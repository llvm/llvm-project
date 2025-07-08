///===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the MustacheHTMLGenerator class,
/// which is Clang-Doc generator for HTML using Mustache templates.
///
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "Representation.h"
#include "support/File.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mustache.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeProfiler.h"

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

namespace clang {
namespace doc {

static Error createFileOpenError(StringRef FileName, std::error_code EC) {
  return createFileError("cannot open file " + FileName, EC);
}

class MustacheHTMLGenerator : public Generator {
public:
  static const char *Format;
  Error generateDocs(StringRef RootDir,
                     StringMap<std::unique_ptr<doc::Info>> Infos,
                     const ClangDocContext &CDCtx) override;
  Error createResources(ClangDocContext &CDCtx) override;
  Error generateDocForInfo(Info *I, raw_ostream &OS,
                           const ClangDocContext &CDCtx) override;
};

class MustacheTemplateFile : public Template {
public:
  static Expected<std::unique_ptr<MustacheTemplateFile>>
  createMustacheFile(StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);

    std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrError.get());
    StringRef FileContent = Buffer->getBuffer();
    return std::make_unique<MustacheTemplateFile>(FileContent);
  }

  Error registerPartialFile(StringRef Name, StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);

    std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrError.get());
    StringRef FileContent = Buffer->getBuffer();
    registerPartial(Name.str(), FileContent.str());
    return Error::success();
  }

  MustacheTemplateFile(StringRef TemplateStr) : Template(TemplateStr) {}
};

static std::unique_ptr<MustacheTemplateFile> NamespaceTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> RecordTemplate = nullptr;

static Error
setupTemplate(std::unique_ptr<MustacheTemplateFile> &Template,
              StringRef TemplatePath,
              std::vector<std::pair<StringRef, StringRef>> Partials) {
  auto T = MustacheTemplateFile::createMustacheFile(TemplatePath);
  if (Error Err = T.takeError())
    return Err;
  Template = std::move(T.get());
  for (const auto &[Name, FileName] : Partials)
    if (auto Err = Template->registerPartialFile(Name, FileName))
      return Err;
  return Error::success();
}

static Error setupTemplateFiles(const clang::doc::ClangDocContext &CDCtx) {
  // Template files need to use the native path when they're opened,
  // but have to be used in POSIX style when used in HTML.
  auto ConvertToNative = [](std::string &&Path) -> std::string {
    SmallString<128> PathBuf(Path);
    llvm::sys::path::native(PathBuf);
    return PathBuf.str().str();
  };

  std::string NamespaceFilePath =
      ConvertToNative(CDCtx.MustacheTemplates.lookup("namespace-template"));
  std::string ClassFilePath =
      ConvertToNative(CDCtx.MustacheTemplates.lookup("class-template"));
  std::string CommentFilePath =
      ConvertToNative(CDCtx.MustacheTemplates.lookup("comment-template"));
  std::string FunctionFilePath =
      ConvertToNative(CDCtx.MustacheTemplates.lookup("function-template"));
  std::string EnumFilePath =
      ConvertToNative(CDCtx.MustacheTemplates.lookup("enum-template"));
  std::vector<std::pair<StringRef, StringRef>> Partials = {
      {"Comments", CommentFilePath},
      {"FunctionPartial", FunctionFilePath},
      {"EnumPartial", EnumFilePath}};

  if (Error Err = setupTemplate(NamespaceTemplate, NamespaceFilePath, Partials))
    return Err;

  if (Error Err = setupTemplate(RecordTemplate, ClassFilePath, Partials))
    return Err;

  return Error::success();
}

Error MustacheHTMLGenerator::generateDocs(
    StringRef RootDir, StringMap<std::unique_ptr<doc::Info>> Infos,
    const clang::doc::ClangDocContext &CDCtx) {
  {
    llvm::TimeTraceScope TS("Setup Templates");
    if (auto Err = setupTemplateFiles(CDCtx))
      return Err;
  }

  // Track which directories we already tried to create.
  StringSet<> CreatedDirs;
  // Collect all output by file name and create the necessary directories.
  StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    llvm::TimeTraceScope TS("setup directories");
    doc::Info *Info = Group.getValue().get();

    SmallString<128> Path;
    sys::path::native(RootDir, Path);
    sys::path::append(Path, Info->getRelativeFilePath(""));
    if (!CreatedDirs.contains(Path)) {
      if (std::error_code EC = sys::fs::create_directories(Path))
        return createStringError(EC, "failed to create directory '%s'.",
                                 Path.c_str());
      CreatedDirs.insert(Path);
    }

    sys::path::append(Path, Info->getFileBaseName() + ".html");
    FileToInfos[Path].push_back(Info);
  }

  {
    llvm::TimeTraceScope TS("Generate Docs");
    for (const auto &Group : FileToInfos) {
      llvm::TimeTraceScope TS("Info to Doc");
      std::error_code FileErr;
      raw_fd_ostream InfoOS(Group.getKey(), FileErr, sys::fs::OF_None);
      if (FileErr)
        return createFileOpenError(Group.getKey(), FileErr);

      for (const auto &Info : Group.getValue())
        if (Error Err = generateDocForInfo(Info, InfoOS, CDCtx))
          return Err;
    }
  }
  return Error::success();
}

static json::Value
extractValue(const Location &L,
             std::optional<StringRef> RepositoryUrl = std::nullopt) {
  Object Obj = Object();
  // TODO: Consider using both Start/End line numbers to improve location report
  Obj.insert({"LineNumber", L.StartLineNumber});
  Obj.insert({"Filename", L.Filename});

  if (!L.IsFileInRootDir || !RepositoryUrl)
    return Obj;
  SmallString<128> FileURL(*RepositoryUrl);
  sys::path::append(FileURL, sys::path::Style::posix, L.Filename);
  FileURL += "#" + std::to_string(L.StartLineNumber);
  Obj.insert({"FileURL", FileURL});

  return Obj;
}

static json::Value extractValue(const Reference &I,
                                StringRef CurrentDirectory) {
  SmallString<64> Path = I.getRelativeFilePath(CurrentDirectory);
  sys::path::append(Path, I.getFileBaseName() + ".html");
  sys::path::native(Path, sys::path::Style::posix);
  Object Obj = Object();
  Obj.insert({"Link", Path});
  Obj.insert({"Name", I.Name});
  Obj.insert({"QualName", I.QualName});
  Obj.insert({"ID", toHex(toStringRef(I.USR))});
  return Obj;
}

static json::Value extractValue(const TypedefInfo &I) {
  // Not Supported
  return nullptr;
}

static json::Value extractValue(const CommentInfo &I) {
  Object Obj = Object();

  json::Value ChildVal = Object();
  Object &Child = *ChildVal.getAsObject();

  json::Value ChildArr = Array();
  auto &CARef = *ChildArr.getAsArray();
  CARef.reserve(I.Children.size());
  for (const auto &C : I.Children)
    CARef.emplace_back(extractValue(*C));

  switch (I.Kind) {
  case CommentKind::CK_TextComment: {
    Obj.insert({commentKindToString(I.Kind), I.Text});
    return Obj;
  }

  case CommentKind::CK_BlockCommandComment: {
    Child.insert({"Command", I.Name});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_InlineCommandComment: {
    json::Value ArgsArr = Array();
    auto &ARef = *ArgsArr.getAsArray();
    ARef.reserve(I.Args.size());
    for (const auto &Arg : I.Args)
      ARef.emplace_back(Arg);
    Child.insert({"Command", I.Name});
    Child.insert({"Args", ArgsArr});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_ParamCommandComment:
  case CommentKind::CK_TParamCommandComment: {
    Child.insert({"ParamName", I.ParamName});
    Child.insert({"Direction", I.Direction});
    Child.insert({"Explicit", I.Explicit});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_VerbatimBlockComment: {
    Child.insert({"Text", I.Text});
    if (!I.CloseName.empty())
      Child.insert({"CloseName", I.CloseName});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_VerbatimBlockLineComment:
  case CommentKind::CK_VerbatimLineComment: {
    Child.insert({"Text", I.Text});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_HTMLStartTagComment: {
    json::Value AttrKeysArray = json::Array();
    json::Value AttrValuesArray = json::Array();
    auto &KeyArr = *AttrKeysArray.getAsArray();
    auto &ValArr = *AttrValuesArray.getAsArray();
    KeyArr.reserve(I.AttrKeys.size());
    ValArr.reserve(I.AttrValues.size());
    for (const auto &K : I.AttrKeys)
      KeyArr.emplace_back(K);
    for (const auto &V : I.AttrValues)
      ValArr.emplace_back(V);
    Child.insert({"Name", I.Name});
    Child.insert({"SelfClosing", I.SelfClosing});
    Child.insert({"AttrKeys", AttrKeysArray});
    Child.insert({"AttrValues", AttrValuesArray});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_HTMLEndTagComment: {
    Child.insert({"Name", I.Name});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_FullComment:
  case CommentKind::CK_ParagraphComment: {
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_Unknown: {
    Obj.insert({commentKindToString(I.Kind), I.Text});
    return Obj;
  }
  }
  llvm_unreachable("Unknown comment kind encountered.");
}

static void maybeInsertLocation(std::optional<Location> Loc,
                                const ClangDocContext &CDCtx, Object &Obj) {
  if (!Loc)
    return;
  Location L = *Loc;
  Obj.insert({"Location", extractValue(L, CDCtx.RepositoryUrl)});
}

static void extractDescriptionFromInfo(ArrayRef<CommentInfo> Descriptions,
                                       json::Object &EnumValObj) {
  if (Descriptions.empty())
    return;
  json::Value DescArr = Array();
  json::Array &DescARef = *DescArr.getAsArray();
  DescARef.reserve(Descriptions.size());
  for (const CommentInfo &Child : Descriptions)
    DescARef.emplace_back(extractValue(Child));
  EnumValObj.insert({"EnumValueComments", DescArr});
}

static json::Value extractValue(const FunctionInfo &I, StringRef ParentInfoDir,
                                const ClangDocContext &CDCtx) {
  Object Obj = Object();
  Obj.insert({"Name", I.Name});
  Obj.insert({"ID", toHex(toStringRef(I.USR))});
  Obj.insert({"Access", getAccessSpelling(I.Access).str()});
  Obj.insert({"ReturnType", extractValue(I.ReturnType.Type, ParentInfoDir)});

  json::Value ParamArr = Array();
  json::Array &ParamARef = *ParamArr.getAsArray();
  ParamARef.reserve(I.Params.size());
  for (const auto Val : enumerate(I.Params)) {
    json::Value V = Object();
    auto &VRef = *V.getAsObject();
    VRef.insert({"Name", Val.value().Name});
    VRef.insert({"Type", Val.value().Type.Name});
    VRef.insert({"End", Val.index() + 1 == I.Params.size()});
    ParamARef.emplace_back(V);
  }
  Obj.insert({"Params", ParamArr});

  maybeInsertLocation(I.DefLoc, CDCtx, Obj);
  return Obj;
}

static json::Value extractValue(const EnumInfo &I,
                                const ClangDocContext &CDCtx) {
  Object Obj = Object();
  std::string EnumType = I.Scoped ? "enum class " : "enum ";
  EnumType += I.Name;
  bool HasComment = llvm::any_of(
      I.Members, [](const EnumValueInfo &M) { return !M.Description.empty(); });
  Obj.insert({"EnumName", EnumType});
  Obj.insert({"HasComment", HasComment});
  Obj.insert({"ID", toHex(toStringRef(I.USR))});
  json::Value EnumArr = Array();
  json::Array &EnumARef = *EnumArr.getAsArray();
  EnumARef.reserve(I.Members.size());
  for (const EnumValueInfo &M : I.Members) {
    json::Value EnumValue = Object();
    auto &EnumValObj = *EnumValue.getAsObject();
    EnumValObj.insert({"Name", M.Name});
    if (!M.ValueExpr.empty())
      EnumValObj.insert({"ValueExpr", M.ValueExpr});
    else
      EnumValObj.insert({"Value", M.Value});

    extractDescriptionFromInfo(M.Description, EnumValObj);
    EnumARef.emplace_back(EnumValue);
  }
  Obj.insert({"EnumValues", EnumArr});

  extractDescriptionFromInfo(I.Description, Obj);
  maybeInsertLocation(I.DefLoc, CDCtx, Obj);

  return Obj;
}

static void extractScopeChildren(const ScopeChildren &S, Object &Obj,
                                 StringRef ParentInfoDir,
                                 const ClangDocContext &CDCtx) {
  json::Value NamespaceArr = Array();
  json::Array &NamespaceARef = *NamespaceArr.getAsArray();
  NamespaceARef.reserve(S.Namespaces.size());
  for (const Reference &Child : S.Namespaces)
    NamespaceARef.emplace_back(extractValue(Child, ParentInfoDir));

  if (!NamespaceARef.empty())
    Obj.insert({"Namespace", Object{{"Links", NamespaceArr}}});

  json::Value RecordArr = Array();
  json::Array &RecordARef = *RecordArr.getAsArray();
  RecordARef.reserve(S.Records.size());
  for (const Reference &Child : S.Records)
    RecordARef.emplace_back(extractValue(Child, ParentInfoDir));

  if (!RecordARef.empty())
    Obj.insert({"Record", Object{{"Links", RecordArr}}});

  json::Value FunctionArr = Array();
  json::Array &FunctionARef = *FunctionArr.getAsArray();
  FunctionARef.reserve(S.Functions.size());

  json::Value PublicFunctionArr = Array();
  json::Array &PublicFunctionARef = *PublicFunctionArr.getAsArray();
  PublicFunctionARef.reserve(S.Functions.size());

  json::Value ProtectedFunctionArr = Array();
  json::Array &ProtectedFunctionARef = *ProtectedFunctionArr.getAsArray();
  ProtectedFunctionARef.reserve(S.Functions.size());

  for (const FunctionInfo &Child : S.Functions) {
    json::Value F = extractValue(Child, ParentInfoDir, CDCtx);
    AccessSpecifier Access = Child.Access;
    if (Access == AccessSpecifier::AS_public)
      PublicFunctionARef.emplace_back(F);
    else if (Access == AccessSpecifier::AS_protected)
      ProtectedFunctionARef.emplace_back(F);
    else
      FunctionARef.emplace_back(F);
  }

  if (!FunctionARef.empty())
    Obj.insert({"Function", Object{{"Obj", FunctionArr}}});

  if (!PublicFunctionARef.empty())
    Obj.insert({"PublicFunction", Object{{"Obj", PublicFunctionArr}}});

  if (!ProtectedFunctionARef.empty())
    Obj.insert({"ProtectedFunction", Object{{"Obj", ProtectedFunctionArr}}});

  json::Value EnumArr = Array();
  auto &EnumARef = *EnumArr.getAsArray();
  EnumARef.reserve(S.Enums.size());
  for (const EnumInfo &Child : S.Enums)
    EnumARef.emplace_back(extractValue(Child, CDCtx));

  if (!EnumARef.empty())
    Obj.insert({"Enums", Object{{"Obj", EnumArr}}});

  json::Value TypedefArr = Array();
  auto &TypedefARef = *TypedefArr.getAsArray();
  TypedefARef.reserve(S.Typedefs.size());
  for (const TypedefInfo &Child : S.Typedefs)
    TypedefARef.emplace_back(extractValue(Child));

  if (!TypedefARef.empty())
    Obj.insert({"Typedefs", Object{{"Obj", TypedefArr}}});
}

static json::Value extractValue(const NamespaceInfo &I,
                                const ClangDocContext &CDCtx) {
  Object NamespaceValue = Object();
  std::string InfoTitle = I.Name.empty() ? "Global Namespace"
                                         : (Twine("namespace ") + I.Name).str();

  SmallString<64> BasePath = I.getRelativeFilePath("");
  NamespaceValue.insert({"NamespaceTitle", InfoTitle});
  NamespaceValue.insert({"NamespacePath", BasePath});

  extractDescriptionFromInfo(I.Description, NamespaceValue);
  extractScopeChildren(I.Children, NamespaceValue, BasePath, CDCtx);
  return NamespaceValue;
}

static json::Value extractValue(const RecordInfo &I,
                                const ClangDocContext &CDCtx) {
  Object RecordValue = Object();
  extractDescriptionFromInfo(I.Description, RecordValue);
  RecordValue.insert({"Name", I.Name});
  RecordValue.insert({"FullName", I.FullName});
  RecordValue.insert({"RecordType", getTagType(I.TagType)});

  maybeInsertLocation(I.DefLoc, CDCtx, RecordValue);

  SmallString<64> BasePath = I.getRelativeFilePath("");
  extractScopeChildren(I.Children, RecordValue, BasePath, CDCtx);
  json::Value PublicMembers = Array();
  json::Array &PubMemberRef = *PublicMembers.getAsArray();
  PubMemberRef.reserve(I.Members.size());
  json::Value ProtectedMembers = Array();
  json::Array &ProtMemberRef = *ProtectedMembers.getAsArray();
  ProtMemberRef.reserve(I.Members.size());
  json::Value PrivateMembers = Array();
  json::Array &PrivMemberRef = *PrivateMembers.getAsArray();
  PrivMemberRef.reserve(I.Members.size());
  for (const MemberTypeInfo &Member : I.Members) {
    json::Value MemberValue = Object();
    auto &MVRef = *MemberValue.getAsObject();
    MVRef.insert({"Name", Member.Name});
    MVRef.insert({"Type", Member.Type.Name});
    extractDescriptionFromInfo(Member.Description, MVRef);

    if (Member.Access == AccessSpecifier::AS_public)
      PubMemberRef.emplace_back(MemberValue);
    else if (Member.Access == AccessSpecifier::AS_protected)
      ProtMemberRef.emplace_back(MemberValue);
    else if (Member.Access == AccessSpecifier::AS_private)
      ProtMemberRef.emplace_back(MemberValue);
  }
  if (!PubMemberRef.empty())
    RecordValue.insert({"PublicMembers", Object{{"Obj", PublicMembers}}});
  if (!ProtMemberRef.empty())
    RecordValue.insert({"ProtectedMembers", Object{{"Obj", ProtectedMembers}}});
  if (!PrivMemberRef.empty())
    RecordValue.insert({"PrivateMembers", Object{{"Obj", PrivateMembers}}});

  return RecordValue;
}

static Error setupTemplateValue(const ClangDocContext &CDCtx, json::Value &V,
                                Info *I) {
  V.getAsObject()->insert({"ProjectName", CDCtx.ProjectName});
  json::Value StylesheetArr = Array();
  auto InfoPath = I->getRelativeFilePath("");
  SmallString<128> RelativePath = computeRelativePath("", InfoPath);
  sys::path::native(RelativePath, sys::path::Style::posix);

  auto *SSA = StylesheetArr.getAsArray();
  SSA->reserve(CDCtx.UserStylesheets.size());
  for (const auto &FilePath : CDCtx.UserStylesheets) {
    SmallString<128> StylesheetPath = RelativePath;
    sys::path::append(StylesheetPath, sys::path::Style::posix,
                      sys::path::filename(FilePath));
    SSA->emplace_back(StylesheetPath);
  }
  V.getAsObject()->insert({"Stylesheets", StylesheetArr});

  json::Value ScriptArr = Array();
  auto *SCA = ScriptArr.getAsArray();
  SCA->reserve(CDCtx.JsScripts.size());
  for (auto Script : CDCtx.JsScripts) {
    SmallString<128> JsPath = RelativePath;
    sys::path::append(JsPath, sys::path::Style::posix,
                      sys::path::filename(Script));
    SCA->emplace_back(JsPath);
  }
  V.getAsObject()->insert({"Scripts", ScriptArr});
  return Error::success();
}

Error MustacheHTMLGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                                const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace: {
    json::Value V =
        extractValue(*static_cast<clang::doc::NamespaceInfo *>(I), CDCtx);
    if (auto Err = setupTemplateValue(CDCtx, V, I))
      return Err;
    assert(NamespaceTemplate && "NamespaceTemplate is nullptr.");
    NamespaceTemplate->render(V, OS);
    break;
  }
  case InfoType::IT_record: {
    json::Value V =
        extractValue(*static_cast<clang::doc::RecordInfo *>(I), CDCtx);
    if (auto Err = setupTemplateValue(CDCtx, V, I))
      return Err;
    // Serialize the JSON value to the output stream in a readable format.
    RecordTemplate->render(V, OS);
    break;
  }
  case InfoType::IT_enum:
    OS << "IT_enum\n";
    break;
  case InfoType::IT_function:
    OS << "IT_Function\n";
    break;
  case InfoType::IT_typedef:
    OS << "IT_typedef\n";
    break;
  case InfoType::IT_concept:
    break;
  case InfoType::IT_variable:
  case InfoType::IT_friend:
    break;
  case InfoType::IT_default:
    return createStringError(inconvertibleErrorCode(), "unexpected InfoType");
  }
  return Error::success();
}

Error MustacheHTMLGenerator::createResources(ClangDocContext &CDCtx) {
  for (const auto &FilePath : CDCtx.UserStylesheets)
    if (Error Err = copyFile(FilePath, CDCtx.OutDirectory))
      return Err;
  for (const auto &FilePath : CDCtx.JsScripts)
    if (Error Err = copyFile(FilePath, CDCtx.OutDirectory))
      return Err;
  return Error::success();
}

const char *MustacheHTMLGenerator::Format = "mustache";

static GeneratorRegistry::Add<MustacheHTMLGenerator>
    MHTML(MustacheHTMLGenerator::Format, "Generator for mustache HTML output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int MHTMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
