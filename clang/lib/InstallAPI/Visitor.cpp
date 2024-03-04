//===- Visitor.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/Visitor.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Basic/Linkage.h"
#include "clang/InstallAPI/Frontend.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"

using namespace llvm;
using namespace llvm::MachO;

namespace clang::installapi {

// Exported NamedDecl needs to have external linkage and
// default visibility from LinkageComputer.
static bool isExported(const NamedDecl *D) {
  auto LV = D->getLinkageAndVisibility();
  return isExternallyVisible(LV.getLinkage()) &&
         (LV.getVisibility() == DefaultVisibility);
}

static bool isInlined(const FunctionDecl *D) {
  bool HasInlineAttribute = false;
  bool NoCXXAttr =
      (!D->getASTContext().getLangOpts().CPlusPlus &&
       !D->getASTContext().getTargetInfo().getCXXABI().isMicrosoft() &&
       !D->hasAttr<DLLExportAttr>());

  // Check all redeclarations to find an inline attribute or keyword.
  for (const auto *RD : D->redecls()) {
    if (!RD->isInlined())
      continue;
    HasInlineAttribute = true;
    if (!(NoCXXAttr || RD->hasAttr<GNUInlineAttr>()))
      continue;
    if (RD->doesThisDeclarationHaveABody() &&
        RD->isInlineDefinitionExternallyVisible())
      return false;
  }

  if (!HasInlineAttribute)
    return false;

  return true;
}

static SymbolFlags getFlags(bool WeakDef, bool ThreadLocal) {
  SymbolFlags Result = SymbolFlags::None;
  if (WeakDef)
    Result |= SymbolFlags::WeakDefined;
  if (ThreadLocal)
    Result |= SymbolFlags::ThreadLocalValue;

  return Result;
}

void InstallAPIVisitor::HandleTranslationUnit(ASTContext &ASTCtx) {
  if (ASTCtx.getDiagnostics().hasErrorOccurred())
    return;

  auto *D = ASTCtx.getTranslationUnitDecl();
  TraverseDecl(D);
}

std::string InstallAPIVisitor::getMangledName(const NamedDecl *D) const {
  SmallString<256> Name;
  if (MC->shouldMangleDeclName(D)) {
    raw_svector_ostream NStream(Name);
    MC->mangleName(D, NStream);
  } else
    Name += D->getNameAsString();

  return getBackendMangledName(Name);
}

std::string InstallAPIVisitor::getBackendMangledName(Twine Name) const {
  SmallString<256> FinalName;
  Mangler::getNameWithPrefix(FinalName, Name, DataLayout(Layout));
  return std::string(FinalName);
}

std::optional<HeaderType>
InstallAPIVisitor::getAccessForDecl(const NamedDecl *D) const {
  SourceLocation Loc = D->getLocation();
  if (Loc.isInvalid())
    return std::nullopt;

  // If the loc refers to a macro expansion, InstallAPI needs to first get the
  // file location of the expansion.
  auto FileLoc = SrcMgr.getFileLoc(Loc);
  FileID ID = SrcMgr.getFileID(FileLoc);
  if (ID.isInvalid())
    return std::nullopt;

  const FileEntry *FE = SrcMgr.getFileEntryForID(ID);
  if (!FE)
    return std::nullopt;

  auto Header = Ctx.findAndRecordFile(FE, PP);
  if (!Header.has_value())
    return std::nullopt;

  HeaderType Access = Header.value();
  assert(Access != HeaderType::Unknown && "unexpected access level for global");
  return Access;
}

/// Check if the interface itself or any of its super classes have an
/// exception attribute. InstallAPI needs to export an additional symbol
/// ("OBJC_EHTYPE_$CLASS_NAME") if any of the classes have the exception
/// attribute.
static bool hasObjCExceptionAttribute(const ObjCInterfaceDecl *D) {
  for (; D != nullptr; D = D->getSuperClass())
    if (D->hasAttr<ObjCExceptionAttr>())
      return true;

  return false;
}
void InstallAPIVisitor::recordObjCInstanceVariables(
    const ASTContext &ASTCtx, ObjCContainerRecord *Record, StringRef SuperClass,
    const llvm::iterator_range<
        DeclContext::specific_decl_iterator<ObjCIvarDecl>>
        Ivars) {
  RecordLinkage Linkage = RecordLinkage::Exported;
  const RecordLinkage ContainerLinkage = Record->getLinkage();
  // If fragile, set to unknown.
  if (ASTCtx.getLangOpts().ObjCRuntime.isFragile())
    Linkage = RecordLinkage::Unknown;
  // Linkage should be inherited from container.
  else if (ContainerLinkage != RecordLinkage::Unknown)
    Linkage = ContainerLinkage;
  for (const auto *IV : Ivars) {
    auto Access = getAccessForDecl(IV);
    if (!Access)
      continue;
    StringRef Name = IV->getName();
    const AvailabilityInfo Avail = AvailabilityInfo::createFromDecl(IV);
    auto AC = IV->getCanonicalAccessControl();
    Ctx.Slice->addObjCIVar(Record, Name, Linkage, Avail, IV, *Access, AC);
  }
}

bool InstallAPIVisitor::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  // Skip forward declaration for classes (@class)
  if (!D->isThisDeclarationADefinition())
    return true;

  // Skip over declarations that access could not be collected for.
  auto Access = getAccessForDecl(D);
  if (!Access)
    return true;

  StringRef Name = D->getObjCRuntimeNameAsString();
  const RecordLinkage Linkage =
      isExported(D) ? RecordLinkage::Exported : RecordLinkage::Internal;
  const AvailabilityInfo Avail = AvailabilityInfo::createFromDecl(D);
  const bool IsEHType =
      (!D->getASTContext().getLangOpts().ObjCRuntime.isFragile() &&
       hasObjCExceptionAttribute(D));

  ObjCInterfaceRecord *Class =
      Ctx.Slice->addObjCInterface(Name, Linkage, Avail, D, *Access, IsEHType);

  // Get base class.
  StringRef SuperClassName;
  if (const auto *SuperClass = D->getSuperClass())
    SuperClassName = SuperClass->getObjCRuntimeNameAsString();

  recordObjCInstanceVariables(D->getASTContext(), Class, SuperClassName,
                              D->ivars());
  return true;
}

bool InstallAPIVisitor::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  StringRef CategoryName = D->getName();
  // Skip over declarations that access could not be collected for.
  auto Access = getAccessForDecl(D);
  if (!Access)
    return true;
  const AvailabilityInfo Avail = AvailabilityInfo::createFromDecl(D);
  const ObjCInterfaceDecl *InterfaceD = D->getClassInterface();
  const StringRef InterfaceName = InterfaceD->getName();

  ObjCCategoryRecord *Category = Ctx.Slice->addObjCCategory(
      InterfaceName, CategoryName, Avail, D, *Access);
  recordObjCInstanceVariables(D->getASTContext(), Category, InterfaceName,
                              D->ivars());
  return true;
}

bool InstallAPIVisitor::VisitVarDecl(const VarDecl *D) {
  // Skip function parameters.
  if (isa<ParmVarDecl>(D))
    return true;

  // Skip variables in records. They are handled seperately for C++.
  if (D->getDeclContext()->isRecord())
    return true;

  // Skip anything inside functions or methods.
  if (!D->isDefinedOutsideFunctionOrMethod())
    return true;

  // If this is a template but not specialization or instantiation, skip.
  if (D->getASTContext().getTemplateOrSpecializationInfo(D) &&
      D->getTemplateSpecializationKind() == TSK_Undeclared)
    return true;

  // Skip over declarations that access could not collected for.
  auto Access = getAccessForDecl(D);
  if (!Access)
    return true;

  const RecordLinkage Linkage =
      isExported(D) ? RecordLinkage::Exported : RecordLinkage::Internal;
  const bool WeakDef = D->hasAttr<WeakAttr>();
  const bool ThreadLocal = D->getTLSKind() != VarDecl::TLS_None;
  const AvailabilityInfo Avail = AvailabilityInfo::createFromDecl(D);
  Ctx.Slice->addGlobal(getMangledName(D), Linkage, GlobalRecord::Kind::Variable,
                       Avail, D, *Access, getFlags(WeakDef, ThreadLocal));
  return true;
}

bool InstallAPIVisitor::VisitFunctionDecl(const FunctionDecl *D) {
  if (const CXXMethodDecl *M = dyn_cast<CXXMethodDecl>(D)) {
    // Skip member function in class templates.
    if (M->getParent()->getDescribedClassTemplate() != nullptr)
      return true;

    // Skip methods in CXX RecordDecls.
    for (auto P : D->getASTContext().getParents(*M)) {
      if (P.get<CXXRecordDecl>())
        return true;
    }

    // Skip CXX ConstructorDecls and DestructorDecls.
    if (isa<CXXConstructorDecl>(M) || isa<CXXDestructorDecl>(M))
      return true;
  }

  // Skip templated functions.
  switch (D->getTemplatedKind()) {
  case FunctionDecl::TK_NonTemplate:
  case FunctionDecl::TK_DependentNonTemplate:
    break;
  case FunctionDecl::TK_MemberSpecialization:
  case FunctionDecl::TK_FunctionTemplateSpecialization:
    if (auto *TempInfo = D->getTemplateSpecializationInfo()) {
      if (!TempInfo->isExplicitInstantiationOrSpecialization())
        return true;
    }
    break;
  case FunctionDecl::TK_FunctionTemplate:
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization:
    return true;
  }

  auto Access = getAccessForDecl(D);
  if (!Access)
    return true;
  auto Name = getMangledName(D);
  const AvailabilityInfo Avail = AvailabilityInfo::createFromDecl(D);
  const bool ExplicitInstantiation = D->getTemplateSpecializationKind() ==
                                     TSK_ExplicitInstantiationDeclaration;
  const bool WeakDef = ExplicitInstantiation || D->hasAttr<WeakAttr>();
  const bool Inlined = isInlined(D);
  const RecordLinkage Linkage = (Inlined || !isExported(D))
                                    ? RecordLinkage::Internal
                                    : RecordLinkage::Exported;
  Ctx.Slice->addGlobal(Name, Linkage, GlobalRecord::Kind::Function, Avail, D,
                       *Access, getFlags(WeakDef, /*ThreadLocal=*/false),
                       Inlined);
  return true;
}

} // namespace clang::installapi
