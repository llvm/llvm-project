//===--- IndexRecordHasher.cpp - Index record hashing ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IndexRecordHasher.h"
#include "FileIndexRecord.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/HashBuilder.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::index;
using namespace llvm;

namespace {

struct IndexRecordHasher {
  ASTContext &Ctx;
  llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
      HashBuilder;

  explicit IndexRecordHasher(ASTContext &context) : Ctx(context) {}

  void hashDecl(const Decl *D);
  void hashMacro(const IdentifierInfo *name, const MacroInfo *macroInfo);

  void hashType(QualType ty);
  void hashCanonicalType(CanQualType canTy);
  void hashDeclName(DeclarationName name);
  void hashNameSpec(const NestedNameSpecifier *nestedName);
  void hashSelector(Selector selector);
  void hashTemplateName(TemplateName name);
  void hashTemplateArg(const TemplateArgument &arg);
  void hashLoc(SourceLocation loc, bool includeOffset);
  void hashAPInt(const APInt &val);
};

class DeclHashVisitor : public ConstDeclVisitor<DeclHashVisitor, void> {
  IndexRecordHasher &Hasher;

public:
  DeclHashVisitor(IndexRecordHasher &Hasher) : Hasher(Hasher) {}

  void VisitDecl(const Decl *decl) {
    return VisitDeclContext(decl->getDeclContext());
  }

  void VisitNamedDecl(const NamedDecl *named) {
    VisitDecl(named);
    if (auto *attr = named->getExternalSourceSymbolAttr()) {
      Hasher.HashBuilder.add(attr->getDefinedIn());
    }
    Hasher.hashDeclName(named->getDeclName());
  }

  void VisitTagDecl(const TagDecl *tag) {
    if (!tag->getDeclName().isEmpty()) {
      Hasher.HashBuilder.add('T');
      VisitTypeDecl(tag);
      return;
    }

    if (const TypedefNameDecl *TD = tag->getTypedefNameForAnonDecl()) {
      Visit(TD);
      return;
    }

    VisitDeclContext(tag->getDeclContext());
    if (tag->isEmbeddedInDeclarator() && !tag->isFreeStanding()) {
      Hasher.hashLoc(tag->getLocation(), /*IncludeOffset=*/true);
      return;
    }

    Hasher.HashBuilder.add('a');
  }

  void VisitClassTemplateSpecializationDecl(
      const ClassTemplateSpecializationDecl *D) {
    Hasher.HashBuilder.add('>');
    VisitCXXRecordDecl(D);
    for (const TemplateArgument &arg : D->getTemplateArgs().asArray()) {
      Hasher.hashTemplateArg(arg);
    }
  }

  void VisitObjCContainerDecl(const ObjCContainerDecl *container) {
    Hasher.HashBuilder.add('I');
    VisitNamedDecl(container);
  }

  void VisitObjCImplDecl(const ObjCImplDecl *impl) {
    if (auto *interface = impl->getClassInterface()) {
      return VisitObjCInterfaceDecl(interface);
    }
  }

  void VisitObjCCategoryDecl(const ObjCCategoryDecl *category) {
    // FIXME: Differentiate between category and the interface ?
    if (auto *interface = category->getClassInterface())
      return VisitObjCInterfaceDecl(interface);
  }

  void VisitFunctionDecl(const FunctionDecl *func) {
    VisitNamedDecl(func);
    if ((!Hasher.Ctx.getLangOpts().CPlusPlus &&
         !func->hasAttr<OverloadableAttr>()) ||
        func->isExternC())
      return;

    for (const auto *param : func->parameters()) {
      Hasher.hashType(param->getType());
    }
  }

  void VisitUnresolvedUsingTypenameDecl(
      const UnresolvedUsingTypenameDecl *unresolved) {
    VisitNamedDecl(unresolved);
    Hasher.hashNameSpec(unresolved->getQualifier());
  }

  void
  VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *unresolved) {
    VisitNamedDecl(unresolved);
    Hasher.hashNameSpec(unresolved->getQualifier());
  }

  void VisitDeclContext(const DeclContext *context) {
    // FIXME: Add location if this is anonymous namespace ?
    context = context->getRedeclContext();
    const Decl *decl = cast<Decl>(context)->getCanonicalDecl();
    if (const auto *named = dyn_cast<NamedDecl>(decl)) {
      Hasher.hashDecl(named);
    }
  }
};
}

std::array<uint8_t, 8> index::hashRecord(const FileIndexRecord &record,
                                         ASTContext &context) {
  IndexRecordHasher hasher(context);

  for (auto &Info : record.getDeclOccurrencesSortedByOffset()) {
    hasher.HashBuilder.add(Info.Roles);
    hasher.HashBuilder.add(Info.Offset);

    if (auto *D = Info.DeclOrMacro.dyn_cast<const Decl *>()) {
      hasher.hashDecl(D);
    } else {
      hasher.hashMacro(Info.MacroName,
                       Info.DeclOrMacro.get<const MacroInfo *>());
    }

    for (auto &Rel : Info.Relations) {
      hasher.hashDecl(Rel.RelatedSymbol);
    }
  }

  return hasher.HashBuilder.final();
}

void IndexRecordHasher::hashMacro(const IdentifierInfo *name,
                                  const MacroInfo *macroInfo) {
  HashBuilder.add("@macro@");
  HashBuilder.add(name->getName());

  auto &sm = Ctx.getSourceManager();
  auto loc = macroInfo->getDefinitionLoc();
  // Only hash the location if it's not in a system header, to match how
  // USR generation behaves.
  if (loc.isValid() && !sm.isInSystemHeader(loc)) {
    hashLoc(loc, /*IncludeOffset=*/true);
  }
}

void IndexRecordHasher::hashDecl(const Decl *D) {
  assert(D->isCanonicalDecl());

  if (auto *NS = dyn_cast<NamespaceDecl>(D)) {
    if (NS->isAnonymousNamespace()) {
      HashBuilder.add("@aN");
      return;
    }
  }

  DeclHashVisitor(*this).Visit(D);
}

void IndexRecordHasher::hashType(QualType NonCanTy) {
  CanQualType CanTy = Ctx.getCanonicalType(NonCanTy);
  hashCanonicalType(CanTy);
}

void IndexRecordHasher::hashCanonicalType(CanQualType CT) {
  auto asCanon = [](QualType Ty) -> CanQualType {
    return CanQualType::CreateUnsafe(Ty);
  };

  while (true) {
    Qualifiers Q = CT.getQualifiers();
    CT = CT.getUnqualifiedType();
    const Type *T = CT.getTypePtr();
    unsigned qVal = 0;
    if (Q.hasConst())
      qVal |= 0x1;
    if (Q.hasVolatile())
      qVal |= 0x2;
    if (Q.hasRestrict())
      qVal |= 0x4;
    if(qVal)
      HashBuilder.add(qVal);

    // Hash in ObjC GC qualifiers?

    if (const BuiltinType *builtin = dyn_cast<BuiltinType>(T)) {
      HashBuilder.add(builtin->getKind());
      return;
    }
    if (const PointerType *pointer = dyn_cast<PointerType>(T)) {
      HashBuilder.add('*');
      CT = asCanon(pointer->getPointeeType());
      continue;
    }
    if (const ReferenceType *ref = dyn_cast<ReferenceType>(T)) {
      HashBuilder.add('&');
      CT = asCanon(ref->getPointeeType());
      continue;
    }
    if (const BlockPointerType *block = dyn_cast<BlockPointerType>(T)) {
      HashBuilder.add('B');
      CT = asCanon(block->getPointeeType());
      continue;
    }
    if (const ObjCObjectPointerType *pointer =
            dyn_cast<ObjCObjectPointerType>(T)) {
      HashBuilder.add('*');
      CT = asCanon(pointer->getPointeeType());
      continue;
    }
    if (const TagType *tag = dyn_cast<TagType>(T)) {
      HashBuilder.add('$');
      hashDecl(tag->getDecl()->getCanonicalDecl());
      return;
    }
    if (const ObjCInterfaceType *interface = dyn_cast<ObjCInterfaceType>(T)) {
      HashBuilder.add('$');
      hashDecl(interface->getDecl()->getCanonicalDecl());
      return;
    }
    if (const ObjCObjectType *obj = dyn_cast<ObjCObjectType>(T)) {
      for (auto *proto : obj->getProtocols()) {
        hashDecl(proto);
      }
      CT = asCanon(obj->getBaseType());
      continue;
    }
    if (const TemplateTypeParmType *param = dyn_cast<TemplateTypeParmType>(T)) {
      HashBuilder.add('t');
      HashBuilder.add(param->getDepth());
      HashBuilder.add(param->getIndex());
      return;
    }
    if (const InjectedClassNameType *injected =
            dyn_cast<InjectedClassNameType>(T)) {
      CT =
          asCanon(injected->getInjectedSpecializationType().getCanonicalType());
      continue;
    }
    if (const PackExpansionType *expansion = dyn_cast<PackExpansionType>(T)) {
      HashBuilder.add('P');
      CT = asCanon(expansion->getPattern());
      continue;
    }
    if (const RValueReferenceType *ref = dyn_cast<RValueReferenceType>(T)) {
      HashBuilder.add('%');
      CT = asCanon(ref->getPointeeType());
      continue;
    }
    if (const FunctionProtoType *func = dyn_cast<FunctionProtoType>(T)) {
      HashBuilder.add('F');
      hashCanonicalType(asCanon(func->getReturnType()));
      for (const auto &paramType : func->param_types()) {
        hashCanonicalType(asCanon(paramType));
      }
      HashBuilder.add(func->isVariadic());
      return;
    }
    if (const ComplexType *complex = dyn_cast<ComplexType>(T)) {
      HashBuilder.add('<');
      CT = asCanon(complex->getElementType());
    }
    if (const TemplateSpecializationType *spec =
            dyn_cast<TemplateSpecializationType>(T)) {
      HashBuilder.add('>');
      hashTemplateName(spec->getTemplateName());
      for (const TemplateArgument &arg : spec->template_arguments()) {
        hashTemplateArg(arg);
      }
      return;
    }
    if (const DependentNameType *depName = dyn_cast<DependentNameType>(T)) {
      HashBuilder.add('^');
      if (const NestedNameSpecifier *nameSpec = depName->getQualifier()) {
        hashNameSpec(nameSpec);
      }
      HashBuilder.add(depName->getIdentifier()->getName());
      return;
    }

    break;
  }
}

void IndexRecordHasher::hashDeclName(DeclarationName name) {
  if (name.isEmpty())
    return;

  HashBuilder.add(name.getNameKind());

  switch (name.getNameKind()) {
  case DeclarationName::Identifier:
    HashBuilder.add(name.getAsIdentifierInfo()->getName());
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    hashSelector(name.getObjCSelector());
    break;
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    break;
  case DeclarationName::CXXOperatorName:
    HashBuilder.add(name.getCXXOverloadedOperator());
    break;
  case DeclarationName::CXXLiteralOperatorName:
    HashBuilder.add(name.getCXXLiteralIdentifier()->getName());
    break;
  case DeclarationName::CXXUsingDirective:
    break;
  case DeclarationName::CXXDeductionGuideName:
    HashBuilder.add(name.getCXXDeductionGuideTemplate()
                        ->getDeclName()
                        .getAsIdentifierInfo()
                        ->getName());
    break;
  }
}

void IndexRecordHasher::hashNameSpec(const NestedNameSpecifier *nameSpec) {
  assert(nameSpec);

  if (auto *prefix = nameSpec->getPrefix()) {
    hashNameSpec(prefix);
  }

  HashBuilder.add(nameSpec->getKind());

  switch (nameSpec->getKind()) {
  case NestedNameSpecifier::Identifier:
    HashBuilder.add(nameSpec->getAsIdentifier()->getName());
    break;

  case NestedNameSpecifier::Namespace:
    hashDecl(nameSpec->getAsNamespace()->getCanonicalDecl());
    break;

  case NestedNameSpecifier::NamespaceAlias:
    hashDecl(nameSpec->getAsNamespaceAlias()->getCanonicalDecl());
    break;

  case NestedNameSpecifier::Global:
    break;

  case NestedNameSpecifier::Super:
    break;

  case NestedNameSpecifier::TypeSpecWithTemplate:
    // Fall through to hash the type.

  case NestedNameSpecifier::TypeSpec:
    hashType(QualType(nameSpec->getAsType(), 0));
    break;
  }
}

void IndexRecordHasher::hashSelector(Selector selector) {
  unsigned numArgs = selector.getNumArgs();
  if (numArgs == 0) {
    ++numArgs;
  }

  for (unsigned i = 0; i < numArgs; ++i) {
    HashBuilder.add(selector.getNameForSlot(i));
  }
}

void IndexRecordHasher::hashTemplateName(TemplateName name) {
  if (TemplateDecl *decl = name.getAsTemplateDecl()) {
    if (TemplateTemplateParmDecl *param =
            dyn_cast<TemplateTemplateParmDecl>(decl)) {
      HashBuilder.add('t');
      HashBuilder.add(param->getDepth());
      HashBuilder.add(param->getIndex());
      return;
    }

    hashDecl(decl->getCanonicalDecl());
  }

  // FIXME: Hash dependent template names.
}

void IndexRecordHasher::hashTemplateArg(const TemplateArgument &arg) {
  switch (arg.getKind()) {
  case TemplateArgument::Null:
    break;

  case TemplateArgument::Declaration:
    hashDecl(arg.getAsDecl());
    break;

  case TemplateArgument::NullPtr:
    break;

  case TemplateArgument::TemplateExpansion:
    HashBuilder.add('P'); // pack expansion of...
    LLVM_FALLTHROUGH;
  case TemplateArgument::Template:
    hashTemplateName(arg.getAsTemplateOrTemplatePattern());
    break;
      
  case TemplateArgument::Expression:
    // FIXME: Hash expressions.
    break;
      
  case TemplateArgument::Pack:
    HashBuilder.add('p');
    for (const auto &element : arg.pack_elements()) {
      hashTemplateArg(element);
    }
    break;
      
  case TemplateArgument::Type:
    hashType(arg.getAsType());
    break;
      
  case TemplateArgument::Integral:
    HashBuilder.add('V');
    hashType(arg.getIntegralType());
    hashAPInt(arg.getAsIntegral());
    break;

  case TemplateArgument::StructuralValue:
    // FIXME: Hash structural values
    break;
  }
}

void IndexRecordHasher::hashLoc(SourceLocation loc, bool includeOffset) {
  if (loc.isInvalid())
    return;

  auto &SM = Ctx.getSourceManager();
  loc = SM.getFileLoc(loc);

  const std::pair<FileID, unsigned> &decomposed = SM.getDecomposedLoc(loc);
  OptionalFileEntryRef entry = SM.getFileEntryRefForID(decomposed.first);
  if (!entry)
    return;

  HashBuilder.add(llvm::sys::path::filename(entry->getName()));

  if (includeOffset) {
    // Use the offest into the FileID to represent the location.  Using
    // a line/column can cause us to look back at the original source file,
    // which is expensive.
    HashBuilder.add(decomposed.second);
  }
}

void IndexRecordHasher::hashAPInt(const APInt &val) {
  HashBuilder.add(val.getBitWidth());
  HashBuilder.addRangeElements(
      llvm::ArrayRef(val.getRawData(), val.getNumWords()));
}
