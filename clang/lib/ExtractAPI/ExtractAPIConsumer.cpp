//===- ExtractAPI/ExtractAPIConsumer.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ExtractAPIAction, and ASTVisitor/Consumer to
/// collect API information.
///
//===----------------------------------------------------------------------===//

#include "TypedefUnderlyingTypeResolver.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "clang/ExtractAPI/FrontendActions.h"
#include "clang/ExtractAPI/Serialization/SymbolGraphSerializer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <utility>

using namespace clang;
using namespace extractapi;

namespace {

StringRef getTypedefName(const TagDecl *Decl) {
  if (const auto *TypedefDecl = Decl->getTypedefNameForAnonDecl())
    return TypedefDecl->getName();

  return {};
}

Optional<std::string> getRelativeIncludeName(const CompilerInstance &CI,
                                             StringRef File,
                                             bool *IsQuoted = nullptr) {
  assert(CI.hasFileManager() &&
         "CompilerInstance does not have a FileNamager!");

  using namespace llvm::sys;
  // Matches framework include patterns
  const llvm::Regex Rule("/(.+)\\.framework/(.+)?Headers/(.+)");

  const auto &FS = CI.getVirtualFileSystem();

  SmallString<128> FilePath(File.begin(), File.end());
  FS.makeAbsolute(FilePath);
  path::remove_dots(FilePath, true);
  FilePath = path::convert_to_slash(FilePath);
  File = FilePath;

  // Checks whether `Dir` is a strict path prefix of `File`. If so returns
  // the prefix length. Otherwise return 0.
  auto CheckDir = [&](llvm::StringRef Dir) -> unsigned {
    llvm::SmallString<32> DirPath(Dir.begin(), Dir.end());
    FS.makeAbsolute(DirPath);
    path::remove_dots(DirPath, true);
    Dir = DirPath;
    for (auto NI = path::begin(File), NE = path::end(File),
              DI = path::begin(Dir), DE = path::end(Dir);
         /*termination condition in loop*/; ++NI, ++DI) {
      // '.' components in File are ignored.
      while (NI != NE && *NI == ".")
        ++NI;
      if (NI == NE)
        break;

      // '.' components in Dir are ignored.
      while (DI != DE && *DI == ".")
        ++DI;

      // Dir is a prefix of File, up to '.' components and choice of path
      // separators.
      if (DI == DE)
        return NI - path::begin(File);

      // Consider all path separators equal.
      if (NI->size() == 1 && DI->size() == 1 &&
          path::is_separator(NI->front()) && path::is_separator(DI->front()))
        continue;

      // Special case Apple .sdk folders since the search path is typically a
      // symlink like `iPhoneSimulator14.5.sdk` while the file is instead
      // located in `iPhoneSimulator.sdk` (the real folder).
      if (NI->endswith(".sdk") && DI->endswith(".sdk")) {
        StringRef NBasename = path::stem(*NI);
        StringRef DBasename = path::stem(*DI);
        if (DBasename.startswith(NBasename))
          continue;
      }

      if (*NI != *DI)
        break;
    }
    return 0;
  };

  unsigned PrefixLength = 0;

  // Go through the search paths and find the first one that is a prefix of
  // the header.
  for (const auto &Entry : CI.getHeaderSearchOpts().UserEntries) {
    // Note whether the match is found in a quoted entry.
    if (IsQuoted)
      *IsQuoted = Entry.Group == frontend::Quoted;

    if (auto EntryFile = CI.getFileManager().getOptionalFileRef(Entry.Path)) {
      if (auto HMap = HeaderMap::Create(*EntryFile, CI.getFileManager())) {
        // If this is a headermap entry, try to reverse lookup the full path
        // for a spelled name before mapping.
        StringRef SpelledFilename = HMap->reverseLookupFilename(File);
        if (!SpelledFilename.empty())
          return SpelledFilename.str();

        // No matching mapping in this headermap, try next search entry.
        continue;
      }
    }

    // Entry is a directory search entry, try to check if it's a prefix of File.
    PrefixLength = CheckDir(Entry.Path);
    if (PrefixLength > 0) {
      // The header is found in a framework path, construct the framework-style
      // include name `<Framework/Header.h>`
      if (Entry.IsFramework) {
        SmallVector<StringRef, 4> Matches;
        Rule.match(File, &Matches);
        // Returned matches are always in stable order.
        if (Matches.size() != 4)
          return None;

        return path::convert_to_slash(
            (Matches[1].drop_front(Matches[1].rfind('/') + 1) + "/" +
             Matches[3])
                .str());
      }

      // The header is found in a normal search path, strip the search path
      // prefix to get an include name.
      return path::convert_to_slash(File.drop_front(PrefixLength));
    }
  }

  // Couldn't determine a include name, use full path instead.
  return None;
}

struct LocationFileChecker {
  bool isLocationInKnownFile(SourceLocation Loc) {
    // If the loc refers to a macro expansion we need to first get the file
    // location of the expansion.
    auto &SM = CI.getSourceManager();
    auto FileLoc = SM.getFileLoc(Loc);
    FileID FID = SM.getFileID(FileLoc);
    if (FID.isInvalid())
      return false;

    const auto *File = SM.getFileEntryForID(FID);
    if (!File)
      return false;

    if (KnownFileEntries.count(File))
      return true;

    if (ExternalFileEntries.count(File))
      return false;

    StringRef FileName = File->tryGetRealPathName().empty()
                             ? File->getName()
                             : File->tryGetRealPathName();

    // Try to reduce the include name the same way we tried to include it.
    bool IsQuoted = false;
    if (auto IncludeName = getRelativeIncludeName(CI, FileName, &IsQuoted))
      if (llvm::any_of(KnownFiles,
                       [&IsQuoted, &IncludeName](const auto &KnownFile) {
                         return KnownFile.first.equals(*IncludeName) &&
                                KnownFile.second == IsQuoted;
                       })) {
        KnownFileEntries.insert(File);
        return true;
      }

    // Record that the file was not found to avoid future reverse lookup for
    // the same file.
    ExternalFileEntries.insert(File);
    return false;
  }

  LocationFileChecker(const CompilerInstance &CI,
                      SmallVector<std::pair<SmallString<32>, bool>> &KnownFiles)
      : CI(CI), KnownFiles(KnownFiles), ExternalFileEntries() {
    for (const auto &KnownFile : KnownFiles)
      if (auto FileEntry = CI.getFileManager().getFile(KnownFile.first))
        KnownFileEntries.insert(*FileEntry);
  }

private:
  const CompilerInstance &CI;
  SmallVector<std::pair<SmallString<32>, bool>> &KnownFiles;
  llvm::DenseSet<const FileEntry *> KnownFileEntries;
  llvm::DenseSet<const FileEntry *> ExternalFileEntries;
};

/// The RecursiveASTVisitor to traverse symbol declarations and collect API
/// information.
class ExtractAPIVisitor : public RecursiveASTVisitor<ExtractAPIVisitor> {
public:
  ExtractAPIVisitor(ASTContext &Context, LocationFileChecker &LCF, APISet &API)
      : Context(Context), API(API), LCF(LCF) {}

  const APISet &getAPI() const { return API; }

  bool VisitVarDecl(const VarDecl *Decl) {
    // Skip function parameters.
    if (isa<ParmVarDecl>(Decl))
      return true;

    // Skip non-global variables in records (struct/union/class).
    if (Decl->getDeclContext()->isRecord())
      return true;

    // Skip local variables inside function or method.
    if (!Decl->isDefinedOutsideFunctionOrMethod())
      return true;

    // If this is a template but not specialization or instantiation, skip.
    if (Decl->getASTContext().getTemplateOrSpecializationInfo(Decl) &&
        Decl->getTemplateSpecializationKind() == TSK_Undeclared)
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    LinkageInfo Linkage = Decl->getLinkageAndVisibility();
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the variable.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForVar(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    // Add the global variable record to the API set.
    API.addGlobalVar(Name, USR, Loc, Availability, Linkage, Comment,
                     Declaration, SubHeading);
    return true;
  }

  bool VisitFunctionDecl(const FunctionDecl *Decl) {
    if (const auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
      // Skip member function in class templates.
      if (Method->getParent()->getDescribedClassTemplate() != nullptr)
        return true;

      // Skip methods in records.
      for (auto P : Context.getParents(*Method)) {
        if (P.get<CXXRecordDecl>())
          return true;
      }

      // Skip ConstructorDecl and DestructorDecl.
      if (isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method))
        return true;
    }

    // Skip templated functions.
    switch (Decl->getTemplatedKind()) {
    case FunctionDecl::TK_NonTemplate:
      break;
    case FunctionDecl::TK_MemberSpecialization:
    case FunctionDecl::TK_FunctionTemplateSpecialization:
      if (auto *TemplateInfo = Decl->getTemplateSpecializationInfo()) {
        if (!TemplateInfo->isExplicitInstantiationOrSpecialization())
          return true;
      }
      break;
    case FunctionDecl::TK_FunctionTemplate:
    case FunctionDecl::TK_DependentFunctionTemplateSpecialization:
      return true;
    }

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    LinkageInfo Linkage = Decl->getLinkageAndVisibility();
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments, sub-heading, and signature of the function.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForFunction(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);
    FunctionSignature Signature =
        DeclarationFragmentsBuilder::getFunctionSignature(Decl);

    // Add the function record to the API set.
    API.addGlobalFunction(Name, USR, Loc, Availability, Linkage, Comment,
                          Declaration, SubHeading, Signature);
    return true;
  }

  bool VisitEnumDecl(const EnumDecl *Decl) {
    if (!Decl->isComplete())
      return true;

    // Skip forward declaration.
    if (!Decl->isThisDeclarationADefinition())
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    std::string NameString = Decl->getQualifiedNameAsString();
    StringRef Name(NameString);
    if (Name.empty())
      Name = getTypedefName(Decl);

    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the enum.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForEnum(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    EnumRecord *EnumRecord =
        API.addEnum(API.copyString(Name), USR, Loc, Availability, Comment,
                    Declaration, SubHeading);

    // Now collect information about the enumerators in this enum.
    recordEnumConstants(EnumRecord, Decl->enumerators());

    return true;
  }

  bool VisitRecordDecl(const RecordDecl *Decl) {
    if (!Decl->isCompleteDefinition())
      return true;

    // Skip C++ structs/classes/unions
    // TODO: support C++ records
    if (isa<CXXRecordDecl>(Decl))
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    if (Name.empty())
      Name = getTypedefName(Decl);
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the struct.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForStruct(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    StructRecord *StructRecord = API.addStruct(
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading);

    // Now collect information about the fields in this struct.
    recordStructFields(StructRecord, Decl->fields());

    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl) {
    // Skip forward declaration for classes (@class)
    if (!Decl->isThisDeclarationADefinition())
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    LinkageInfo Linkage = Decl->getLinkageAndVisibility();
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the interface.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForObjCInterface(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    // Collect super class information.
    SymbolReference SuperClass;
    if (const auto *SuperClassDecl = Decl->getSuperClass()) {
      SuperClass.Name = SuperClassDecl->getObjCRuntimeNameAsString();
      SuperClass.USR = API.recordUSR(SuperClassDecl);
    }

    ObjCInterfaceRecord *ObjCInterfaceRecord =
        API.addObjCInterface(Name, USR, Loc, Availability, Linkage, Comment,
                             Declaration, SubHeading, SuperClass);

    // Record all methods (selectors). This doesn't include automatically
    // synthesized property methods.
    recordObjCMethods(ObjCInterfaceRecord, Decl->methods());
    recordObjCProperties(ObjCInterfaceRecord, Decl->properties());
    recordObjCInstanceVariables(ObjCInterfaceRecord, Decl->ivars());
    recordObjCProtocols(ObjCInterfaceRecord, Decl->protocols());

    return true;
  }

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl) {
    // Skip forward declaration for protocols (@protocol).
    if (!Decl->isThisDeclarationADefinition())
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the protocol.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForObjCProtocol(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    ObjCProtocolRecord *ObjCProtocolRecord = API.addObjCProtocol(
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading);

    recordObjCMethods(ObjCProtocolRecord, Decl->methods());
    recordObjCProperties(ObjCProtocolRecord, Decl->properties());
    recordObjCProtocols(ObjCProtocolRecord, Decl->protocols());

    return true;
  }

  bool VisitTypedefNameDecl(const TypedefNameDecl *Decl) {
    // Skip ObjC Type Parameter for now.
    if (isa<ObjCTypeParamDecl>(Decl))
      return true;

    if (!Decl->isDefinedOutsideFunctionOrMethod())
      return true;

    if (!LCF.isLocationInKnownFile(Decl->getLocation()))
      return true;

    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    StringRef Name = Decl->getName();
    AvailabilityInfo Availability = getAvailability(Decl);
    StringRef USR = API.recordUSR(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    QualType Type = Decl->getUnderlyingType();
    SymbolReference SymRef =
        TypedefUnderlyingTypeResolver(Context).getSymbolReferenceForType(Type,
                                                                         API);

    API.addTypedef(Name, USR, Loc, Availability, Comment,
                   DeclarationFragmentsBuilder::getFragmentsForTypedef(Decl),
                   DeclarationFragmentsBuilder::getSubHeading(Decl), SymRef);

    return true;
  }

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl) {
    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());
    // Build declaration fragments and sub-heading for the category.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForObjCCategory(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    const ObjCInterfaceDecl *InterfaceDecl = Decl->getClassInterface();
    SymbolReference Interface(InterfaceDecl->getName(),
                              API.recordUSR(InterfaceDecl));

    ObjCCategoryRecord *ObjCCategoryRecord =
        API.addObjCCategory(Name, USR, Loc, Availability, Comment, Declaration,
                            SubHeading, Interface);

    recordObjCMethods(ObjCCategoryRecord, Decl->methods());
    recordObjCProperties(ObjCCategoryRecord, Decl->properties());
    recordObjCInstanceVariables(ObjCCategoryRecord, Decl->ivars());
    recordObjCProtocols(ObjCCategoryRecord, Decl->protocols());

    return true;
  }

private:
  /// Get availability information of the declaration \p D.
  AvailabilityInfo getAvailability(const Decl *D) const {
    StringRef PlatformName = Context.getTargetInfo().getPlatformName();

    AvailabilityInfo Availability;
    // Collect availability attributes from all redeclarations.
    for (const auto *RD : D->redecls()) {
      for (const auto *A : RD->specific_attrs<AvailabilityAttr>()) {
        if (A->getPlatform()->getName() != PlatformName)
          continue;
        Availability = AvailabilityInfo(A->getIntroduced(), A->getDeprecated(),
                                        A->getObsoleted(), A->getUnavailable(),
                                        /* UnconditionallyDeprecated */ false,
                                        /* UnconditionallyUnavailable */ false);
        break;
      }

      if (const auto *A = RD->getAttr<UnavailableAttr>())
        if (!A->isImplicit()) {
          Availability.Unavailable = true;
          Availability.UnconditionallyUnavailable = true;
        }

      if (const auto *A = RD->getAttr<DeprecatedAttr>())
        if (!A->isImplicit())
          Availability.UnconditionallyDeprecated = true;
    }

    return Availability;
  }

  /// Collect API information for the enum constants and associate with the
  /// parent enum.
  void recordEnumConstants(EnumRecord *EnumRecord,
                           const EnumDecl::enumerator_range Constants) {
    for (const auto *Constant : Constants) {
      // Collect symbol information.
      StringRef Name = Constant->getName();
      StringRef USR = API.recordUSR(Constant);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Constant->getLocation());
      AvailabilityInfo Availability = getAvailability(Constant);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Constant))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the enum constant.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForEnumConstant(Constant);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Constant);

      API.addEnumConstant(EnumRecord, Name, USR, Loc, Availability, Comment,
                          Declaration, SubHeading);
    }
  }

  /// Collect API information for the struct fields and associate with the
  /// parent struct.
  void recordStructFields(StructRecord *StructRecord,
                          const RecordDecl::field_range Fields) {
    for (const auto *Field : Fields) {
      // Collect symbol information.
      StringRef Name = Field->getName();
      StringRef USR = API.recordUSR(Field);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Field->getLocation());
      AvailabilityInfo Availability = getAvailability(Field);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Field))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the struct field.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForField(Field);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Field);

      API.addStructField(StructRecord, Name, USR, Loc, Availability, Comment,
                         Declaration, SubHeading);
    }
  }

  /// Collect API information for the Objective-C methods and associate with the
  /// parent container.
  void recordObjCMethods(ObjCContainerRecord *Container,
                         const ObjCContainerDecl::method_range Methods) {
    for (const auto *Method : Methods) {
      // Don't record selectors for properties.
      if (Method->isPropertyAccessor())
        continue;

      StringRef Name = API.copyString(Method->getSelector().getAsString());
      StringRef USR = API.recordUSR(Method);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Method->getLocation());
      AvailabilityInfo Availability = getAvailability(Method);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Method))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments, sub-heading, and signature for the method.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForObjCMethod(Method);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Method);
      FunctionSignature Signature =
          DeclarationFragmentsBuilder::getFunctionSignature(Method);

      API.addObjCMethod(Container, Name, USR, Loc, Availability, Comment,
                        Declaration, SubHeading, Signature,
                        Method->isInstanceMethod());
    }
  }

  void recordObjCProperties(ObjCContainerRecord *Container,
                            const ObjCContainerDecl::prop_range Properties) {
    for (const auto *Property : Properties) {
      StringRef Name = Property->getName();
      StringRef USR = API.recordUSR(Property);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Property->getLocation());
      AvailabilityInfo Availability = getAvailability(Property);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Property))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the property.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForObjCProperty(Property);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Property);

      StringRef GetterName =
          API.copyString(Property->getGetterName().getAsString());
      StringRef SetterName =
          API.copyString(Property->getSetterName().getAsString());

      // Get the attributes for property.
      unsigned Attributes = ObjCPropertyRecord::NoAttr;
      if (Property->getPropertyAttributes() &
          ObjCPropertyAttribute::kind_readonly)
        Attributes |= ObjCPropertyRecord::ReadOnly;
      if (Property->getPropertyAttributes() & ObjCPropertyAttribute::kind_class)
        Attributes |= ObjCPropertyRecord::Class;

      API.addObjCProperty(
          Container, Name, USR, Loc, Availability, Comment, Declaration,
          SubHeading,
          static_cast<ObjCPropertyRecord::AttributeKind>(Attributes),
          GetterName, SetterName, Property->isOptional());
    }
  }

  void recordObjCInstanceVariables(
      ObjCContainerRecord *Container,
      const llvm::iterator_range<
          DeclContext::specific_decl_iterator<ObjCIvarDecl>>
          Ivars) {
    for (const auto *Ivar : Ivars) {
      StringRef Name = Ivar->getName();
      StringRef USR = API.recordUSR(Ivar);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Ivar->getLocation());
      AvailabilityInfo Availability = getAvailability(Ivar);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Ivar))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the instance variable.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForField(Ivar);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Ivar);

      ObjCInstanceVariableRecord::AccessControl Access =
          Ivar->getCanonicalAccessControl();

      API.addObjCInstanceVariable(Container, Name, USR, Loc, Availability,
                                  Comment, Declaration, SubHeading, Access);
    }
  }

  void recordObjCProtocols(ObjCContainerRecord *Container,
                           ObjCInterfaceDecl::protocol_range Protocols) {
    for (const auto *Protocol : Protocols)
      Container->Protocols.emplace_back(Protocol->getName(),
                                        API.recordUSR(Protocol));
  }

  ASTContext &Context;
  APISet &API;
  LocationFileChecker &LCF;
};

class ExtractAPIConsumer : public ASTConsumer {
public:
  ExtractAPIConsumer(ASTContext &Context,
                     std::unique_ptr<LocationFileChecker> LCF, APISet &API)
      : Visitor(Context, *LCF, API), LCF(std::move(LCF)) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    // Use ExtractAPIVisitor to traverse symbol declarations in the context.
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  ExtractAPIVisitor Visitor;
  std::unique_ptr<LocationFileChecker> LCF;
};

class MacroCallback : public PPCallbacks {
public:
  MacroCallback(const SourceManager &SM, LocationFileChecker &LCF, APISet &API,
                Preprocessor &PP)
      : SM(SM), LCF(LCF), API(API), PP(PP) {}

  void MacroDefined(const Token &MacroNameToken,
                    const MacroDirective *MD) override {
    auto *MacroInfo = MD->getMacroInfo();

    if (MacroInfo->isBuiltinMacro())
      return;

    auto SourceLoc = MacroNameToken.getLocation();
    if (SM.isWrittenInBuiltinFile(SourceLoc) ||
        SM.isWrittenInCommandLineFile(SourceLoc))
      return;

    PendingMacros.emplace_back(MacroNameToken, MD);
  }

  // If a macro gets undefined at some point during preprocessing of the inputs
  // it means that it isn't an exposed API and we should therefore not add a
  // macro definition for it.
  void MacroUndefined(const Token &MacroNameToken, const MacroDefinition &MD,
                      const MacroDirective *Undef) override {
    // If this macro wasn't previously defined we don't need to do anything
    // here.
    if (!Undef)
      return;

    llvm::erase_if(PendingMacros, [&MD, this](const PendingMacro &PM) {
      return MD.getMacroInfo()->isIdenticalTo(*PM.MD->getMacroInfo(), PP,
                                              /*Syntactically*/ false);
    });
  }

  void EndOfMainFile() override {
    for (auto &PM : PendingMacros) {
      // `isUsedForHeaderGuard` is only set when the preprocessor leaves the
      // file so check for it here.
      if (PM.MD->getMacroInfo()->isUsedForHeaderGuard())
        continue;

      if (!LCF.isLocationInKnownFile(PM.MacroNameToken.getLocation()))
        continue;

      StringRef Name = PM.MacroNameToken.getIdentifierInfo()->getName();
      PresumedLoc Loc = SM.getPresumedLoc(PM.MacroNameToken.getLocation());
      StringRef USR =
          API.recordUSRForMacro(Name, PM.MacroNameToken.getLocation(), SM);

      API.addMacroDefinition(
          Name, USR, Loc,
          DeclarationFragmentsBuilder::getFragmentsForMacro(Name, PM.MD),
          DeclarationFragmentsBuilder::getSubHeadingForMacro(Name));
    }

    PendingMacros.clear();
  }

private:
  struct PendingMacro {
    Token MacroNameToken;
    const MacroDirective *MD;

    PendingMacro(const Token &MacroNameToken, const MacroDirective *MD)
        : MacroNameToken(MacroNameToken), MD(MD) {}
  };

  const SourceManager &SM;
  LocationFileChecker &LCF;
  APISet &API;
  Preprocessor &PP;
  llvm::SmallVector<PendingMacro> PendingMacros;
};

} // namespace

std::unique_ptr<ASTConsumer>
ExtractAPIAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  OS = CreateOutputFile(CI, InFile);
  if (!OS)
    return nullptr;

  ProductName = CI.getFrontendOpts().ProductName;

  // Now that we have enough information about the language options and the
  // target triple, let's create the APISet before anyone uses it.
  API = std::make_unique<APISet>(
      CI.getTarget().getTriple(),
      CI.getFrontendOpts().Inputs.back().getKind().getLanguage());

  auto LCF = std::make_unique<LocationFileChecker>(CI, KnownInputFiles);

  CI.getPreprocessor().addPPCallbacks(std::make_unique<MacroCallback>(
      CI.getSourceManager(), *LCF, *API, CI.getPreprocessor()));

  return std::make_unique<ExtractAPIConsumer>(CI.getASTContext(),
                                              std::move(LCF), *API);
}

bool ExtractAPIAction::PrepareToExecuteAction(CompilerInstance &CI) {
  auto &Inputs = CI.getFrontendOpts().Inputs;
  if (Inputs.empty())
    return true;

  if (!CI.hasFileManager())
    if (!CI.createFileManager())
      return false;

  auto Kind = Inputs[0].getKind();

  // Convert the header file inputs into a single input buffer.
  SmallString<256> HeaderContents;
  bool IsQuoted = false;
  for (const FrontendInputFile &FIF : Inputs) {
    if (Kind.isObjectiveC())
      HeaderContents += "#import";
    else
      HeaderContents += "#include";

    StringRef FilePath = FIF.getFile();
    if (auto RelativeName = getRelativeIncludeName(CI, FilePath, &IsQuoted)) {
      if (IsQuoted)
        HeaderContents += " \"";
      else
        HeaderContents += " <";

      HeaderContents += *RelativeName;

      if (IsQuoted)
        HeaderContents += "\"\n";
      else
        HeaderContents += ">\n";
      KnownInputFiles.emplace_back(static_cast<SmallString<32>>(*RelativeName),
                                   IsQuoted);
    } else {
      HeaderContents += " \"";
      HeaderContents += FilePath;
      HeaderContents += "\"\n";
      KnownInputFiles.emplace_back(FilePath, true);
    }
  }

  if (CI.getHeaderSearchOpts().Verbose)
    CI.getVerboseOutputStream() << getInputBufferName() << ":\n"
                                << HeaderContents << "\n";

  Buffer = llvm::MemoryBuffer::getMemBufferCopy(HeaderContents,
                                                getInputBufferName());

  // Set that buffer up as our "real" input in the CompilerInstance.
  Inputs.clear();
  Inputs.emplace_back(Buffer->getMemBufferRef(), Kind, /*IsSystem*/ false);

  return true;
}

void ExtractAPIAction::EndSourceFileAction() {
  if (!OS)
    return;

  // Setup a SymbolGraphSerializer to write out collected API information in
  // the Symbol Graph format.
  // FIXME: Make the kind of APISerializer configurable.
  SymbolGraphSerializer SGSerializer(*API, ProductName);
  SGSerializer.serialize(*OS);
  OS.reset();
}

std::unique_ptr<raw_pwrite_stream>
ExtractAPIAction::CreateOutputFile(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<raw_pwrite_stream> OS =
      CI.createDefaultOutputFile(/*Binary=*/false, InFile, /*Extension=*/"json",
                                 /*RemoveFileOnSignal=*/false);
  if (!OS)
    return nullptr;
  return OS;
}
