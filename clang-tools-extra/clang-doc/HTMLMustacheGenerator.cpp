//===-- HTMLMustacheGenerator.cpp - HTML Mustache Generator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Generators.h"
#include "Representation.h"
#include "FileHelpersClangDoc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mustache.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

namespace clang {
namespace doc {

static StringMap<llvm::SmallString<16>> Index;

void generateIndex(Info *I) {
  switch (I->IT) {
  case InfoType::IT_namespace: {
    const NamespaceInfo& N = *static_cast<clang::doc::NamespaceInfo *>(I);
    std::string HexId = llvm::toHex(llvm::toStringRef(N.USR));
    Index[HexId] = HexId;
    for (const EnumInfo& E : N.Children.Enums)
      Index[llvm::toHex(llvm::toStringRef(E.USR))] = HexId;
    for (const TypedefInfo& T : N.Children.Typedefs)
      Index[llvm::toHex(llvm::toStringRef(T.USR))] = HexId;
    break;
  }
  case InfoType::IT_record: {
    const RecordInfo& R = *static_cast<clang::doc::RecordInfo *>(I);
    std::string HexId = llvm::toHex(llvm::toStringRef(R.USR));
    Index[HexId] = HexId;
    for (const EnumInfo& E : R.Children.Enums)
      Index[llvm::toHex(llvm::toStringRef(E.USR))] = HexId;
    for (const TypedefInfo& T : R.Children.Typedefs)
      Index[llvm::toHex(llvm::toStringRef(T.USR))] = HexId;
    break;
  }  
  case InfoType::IT_enum:
    break;
  case InfoType::IT_function:
    break;
  case InfoType::IT_typedef:
    break;
  case InfoType::IT_default:
    return;
  }
}

class MustacheHTMLGenerator : public Generator {
public:
  static const char *Format;
  llvm::Error generateDocs(StringRef RootDir,
                           llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                           const ClangDocContext &CDCtx) override;
  llvm::Error createResources(ClangDocContext &CDCtx) override;
  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                 const ClangDocContext &CDCtx) override;
  
};

class MustacheTemplateFile : public Template {
public:
  static ErrorOr<std::unique_ptr<MustacheTemplateFile>> createMustacheFile
      (StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    
    if (auto EC = BufferOrError.getError()) {
      return EC;
    }
    std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(BufferOrError.get());
    llvm::StringRef FileContent = Buffer->getBuffer();
    return std::make_unique<MustacheTemplateFile>(FileContent);
  }
  
  Error registerPartialFile(StringRef Name, StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return llvm::createFileError("cannot open file", EC);
    std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(BufferOrError.get());
    llvm::StringRef FileContent = Buffer->getBuffer();
    registerPartial(Name, FileContent);
    return llvm::Error::success();
  }

  MustacheTemplateFile(StringRef TemplateStr) : Template(TemplateStr) {}
};

static std::unique_ptr<MustacheTemplateFile> NamespaceTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> RecordTemplate = nullptr;


llvm::Error setupTemplate(
    std::unique_ptr<MustacheTemplateFile> &Template,
    StringRef TemplatePath,
    std::vector<std::pair<StringRef, StringRef>> Partials) {
  auto T = MustacheTemplateFile::createMustacheFile(TemplatePath);
  if (auto EC = T.getError())
    return llvm::createFileError("cannot open file", EC);
  Template = std::move(T.get());
  for (const auto &P : Partials) {
    auto Err = Template->registerPartialFile(P.first, P.second);
    if (Err)
      return Err;
  }
  return llvm::Error::success();
}

llvm::Error 
setupTemplateFiles(const clang::doc::ClangDocContext &CDCtx) {
  auto NamespaceFilePath = CDCtx.MustacheTemplates.lookup("namespace-template");
  auto ClassFilePath = CDCtx.MustacheTemplates.lookup("class-template");
  auto CommentFilePath = CDCtx.MustacheTemplates.lookup("comments-template");
  auto FunctionFilePath = CDCtx.MustacheTemplates.lookup("function-template");
  auto EnumFilePath = CDCtx.MustacheTemplates.lookup("enum-template");
  auto TypeDefFilePath = CDCtx.MustacheTemplates.lookup("typedef-template");
  std::vector<std::pair<StringRef, StringRef>> Partials = {
      {"Comments", CommentFilePath},
      {"FunctionPartial", FunctionFilePath},
      {"EnumPartial", EnumFilePath},
      {"TypedefPartial", TypeDefFilePath}
  };
  
  auto Err = setupTemplate(NamespaceTemplate, NamespaceFilePath, Partials);
  if (Err)
    return Err;
  
  Err = setupTemplate(RecordTemplate, ClassFilePath, Partials);
  
  if (Err)
    return Err;
  
  return llvm::Error::success();
}


llvm::Error 
MustacheHTMLGenerator::generateDocs(llvm::StringRef RootDir, 
                                    llvm::StringMap<std::unique_ptr<doc::Info>> Infos, 
                                    const clang::doc::ClangDocContext &CDCtx) {
  if (auto Err = setupTemplateFiles(CDCtx))
    return Err;
  // Collect all output by file name and create the necessary directories.
  llvm::StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    doc::Info *Info = Group.getValue().get();
    llvm::SmallString<128> Path;
    llvm::sys::path::native(RootDir, Path);
    std::string InfoFile = llvm::toHex(llvm::toStringRef(Info->USR)) + ".html";
    llvm::sys::path::append(Path, InfoFile);
    generateIndex(Info);
    FileToInfos[Path].push_back(Info);
  }
  
  for (const auto &Group : FileToInfos) {
    std::error_code FileErr;
    llvm::raw_fd_ostream InfoOS(Group.getKey(), FileErr,
                                llvm::sys::fs::OF_None);
    if (FileErr)
      return llvm::createStringError(FileErr, "Error opening file '%s'",
                                     Group.getKey().str().c_str());
    
    for (const auto &Info : Group.getValue()) {
      if (llvm::Error Err = generateDocForInfo(Info, InfoOS, CDCtx))
        return Err;
    }
  }
  return llvm::Error::success();
}

Value extractValue(const CommentInfo &I) {
  Object Obj = Object();
  Value Child = Object();
  
  if (I.Kind == "FullComment") {
    Value ChildArr = Array();
    for (const auto& C: I.Children)
      ChildArr.getAsArray()->emplace_back(extractValue(*C));
    Child.getAsObject()->insert({"Children", ChildArr});
    Obj.insert({"FullComment", Child});
  }
  if (I.Kind == "ParagraphComment") {
    Value ChildArr = Array();
    for (const auto& C: I.Children)
      ChildArr.getAsArray()->emplace_back(extractValue(*C));
    Child.getAsObject()->insert({"Children", ChildArr});
    Obj.insert({"ParagraphComment", Child});
  }
  if (I.Kind == "BlockCommandComment") {
    Child.getAsObject()->insert({"Command", I.Name});
    Value ChildArr = Array();
    
    for (const auto& C: I.Children)
      ChildArr.getAsArray()->emplace_back(extractValue(*C));
    Child.getAsObject()->insert({"Children", ChildArr});
    Obj.insert({"BlockCommandComment", Child});
  }
  if (I.Kind == "TextComment")
    Obj.insert({"TextComment", I.Text});
  
  return Obj;
}


Value extractValue(const std::optional<Location> &Loc, 
                   std::optional<StringRef> RepositoryUrl = std::nullopt) {
  
  Object Obj = Object();
  if (Loc.has_value()) {
    Location L = *Loc;
    SmallString<128> Filename(llvm::sys::path::filename(L.Filename));
    Obj.insert({"LineNumber", L.LineNumber});
    Obj.insert({"Filename", Filename});
    if (!RepositoryUrl)
      return Obj;
    // This link only works specifically for github
    SmallString<128> FileURL(*RepositoryUrl);
    llvm::sys::path::append(FileURL, llvm::sys::path::Style::posix, "blob/main");
    llvm::sys::path::append(FileURL, llvm::sys::path::Style::posix, L.Filename);
    FileURL += "#L" + std::to_string(L.LineNumber);
    Obj.insert({"FileURL", FileURL});
  }
  return Obj;
}

Value extractValue(const Reference &I, StringRef CurrentDirectory) {
  llvm::SmallString<64> Path = I.getRelativeFilePath(CurrentDirectory);
  llvm::sys::path::append(Path, I.getFileBaseName() + ".html");
  llvm::sys::path::native(Path, llvm::sys::path::Style::posix);
  Object Obj = Object();
  Obj.insert({"Link", Path});
  Obj.insert({"Name", I.Name});
  Obj.insert({"QualName", I.QualName});
  Obj.insert({"ID", llvm::toHex(llvm::toStringRef(I.USR))});
  return Obj;
}


Value extractValue(const TypedefInfo &I, const ClangDocContext& CDCtx) {
  Object Obj = Object();
  Obj.insert({"ID", llvm::toHex(llvm::toStringRef(I.USR))});
  Obj.insert({"TypeDeclaration", I.TypeDeclaration});
  Obj.insert({"Name", I.Name});
  Obj.insert({"IsAlias", I.IsUsing});
  Obj.insert({"Location", extractValue(I.DefLoc, CDCtx.RepositoryUrl)});
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    Obj.insert({"TypeDefComments", ArrDesc});
  }
  return Obj;
}

// Function to replace a substring within a SmallString with another SmallString
void replaceSubstring(llvm::SmallString<256> &Input, 
                      const llvm::SmallString<16> &From, 
                      const llvm::SmallString<16> &To) {
  llvm::StringRef InputStr = Input;
  llvm::SmallString<16> FromStr;
  raw_svector_ostream Stream(FromStr);
  printHTMLEscaped(From, Stream);
  llvm::StringRef ToStr = To;
  
  // Find the first occurrence of 'from' in 'input'
  size_t Pos = InputStr.find(FromStr);
  while (Pos != llvm::StringRef::npos) {
    // Create a new SmallString to hold the modified string
    llvm::SmallString<256> NewString;

    // Append the part before the found substring
    NewString.append(Input.begin(), Input.begin() + Pos);
        
    // Append the replacement substring
    NewString.append(ToStr.begin(), ToStr.end());
        
    // Append the part after the found substring
    NewString.append(Input.begin() + Pos + FromStr.size(), Input.end());

    // Update 'input' with the modified string
    Input = NewString;

    // Update the input string and find the next occurrence
    InputStr = Input;
    Pos = InputStr.find(FromStr, Pos + ToStr.size());
  }
}

SmallString<64> extractLink(const Reference& R) {
  std::string HexId = llvm::toHex(llvm::toStringRef(R.USR));
  SmallString<16> Name = Index[HexId];
  SmallString<64> Link({Name, ".html"});
  return Link;
}

Value extractRecordPrototype(const SmallString<16> Prototype,
                             llvm::SmallVector<Reference, 4> Parents,
                             llvm::SmallVector<Reference, 4> VirtualParents)
{
  SmallString<256> Result;
  raw_svector_ostream Stream(Result);
  printHTMLEscaped(Prototype, Stream);
  
  for (Reference& R : Parents) 
  {
    SmallString<16> ParentLink({
        "<a style=\"color: #08637D;\" class=\"code-highlight\"",
        " href=\"",extractLink(R),"\">", 
        R.Name,
        "</a>"
    });
    replaceSubstring(Result, R.Name, ParentLink);
  }
  
  for (Reference& V : VirtualParents) 
  {
    SmallString<16> ParentLink({
        "<a style=\"color: #08637D;\" class=\"code-highlight\"",
        " href=\"",extractLink(V),"\">", 
        V.Name,
        "</a>"
    });
    replaceSubstring(Result, V.Name, ParentLink);
  }
  
  return Result;
}

// extract the prototype and adds anchor tags to it
Value extractFunctionPrototype(const SmallString<256>& Prototype,
                       const llvm::SmallVector<FieldTypeInfo, 4>& Params,
                       const TypeInfo& ReturnType) {
  SmallString<256> Result;
  raw_svector_ostream Stream(Result);
  printHTMLEscaped(Prototype, Stream);
  
  if (!ReturnType.IsBuiltIn && !ReturnType.IsTemplate) {
    replaceSubstring(Result, ReturnType.Type.Name, {
       "<a style=\"color: #08637D;\" class=\"code-highlight\"",
       " href=\"",extractLink(ReturnType.Type),"\">", 
       ReturnType.Type.Name, 
      "</a>"
    });
  }
  SmallSet<StringRef, 16> ParamNames;
  for (const FieldTypeInfo& F : Params) 
  {
    if (ParamNames.count(F.Type.Name) > 0 || F.IsBuiltIn || F.IsTemplate)
      continue;
    
    ParamNames.insert(F.Type.Name);
    SmallString<16> ParamLink({
        "<a style=\"color: #08637D;\" class=\"code-highlight\"",
        " href=\"",extractLink(F.Type),"\">", 
        F.Type.Name,
        "</a>"
    });
    replaceSubstring(Result, F.Type.Name, ParamLink);
  }
  return Result;
}

Value extractValue(const FunctionInfo &I, StringRef ParentInfoDir,
                   const ClangDocContext &CDCtx) {
  Object Obj = Object();
  Obj.insert({"Name", I.Name});
  Obj.insert({"FunctionPrototype",
              extractFunctionPrototype(I.ProtoType, I.Params, I.ReturnType)});
  Obj.insert({"ID", llvm::toHex(llvm::toStringRef(I.USR))});
  Obj.insert({"Access", getAccessSpelling(I.Access).str()});
  Obj.insert({"ReturnType", extractValue(I.ReturnType.Type, ParentInfoDir)});
  Value ParamArr = Array();
  for (const auto Val : llvm::enumerate(I.Params)) {
    Value V = Object();
    V.getAsObject()->insert({"Name", Val.value().Name});
    V.getAsObject()->insert({"Type", Val.value().Type.Name});
    V.getAsObject()->insert({"End",  Val.index() + 1 == I.Params.size()});
    ParamArr.getAsArray()->emplace_back(V);
  }
  Obj.insert({"Params", ParamArr});
  
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    Obj.insert({"FunctionComments", ArrDesc});
  }
  Obj.insert({"Location", extractValue(I.DefLoc, CDCtx.RepositoryUrl)});
  return Obj;
}

Value extractValue(const EnumInfo &I, const ClangDocContext &CDCtx) {
  Object Obj = Object();
  std::string EnumType = I.Scoped ? "enum class " : "enum ";
  EnumType += I.Name;
  bool HasComment = std::any_of(
      I.Members.begin(), I.Members.end(),
      [](const EnumValueInfo &M) { return !M.Description.empty(); });
  Obj.insert({"EnumName", EnumType});
  Obj.insert({"HasComment", HasComment});
  Obj.insert({"ID", llvm::toHex(llvm::toStringRef(I.USR))});
  Value Arr = Array();
  for (const EnumValueInfo& M: I.Members) {
    Value EnumValue = Object();
    EnumValue.getAsObject()->insert({"Name", M.Name});
    if (!M.ValueExpr.empty())
      EnumValue.getAsObject()->insert({"ValueExpr", M.ValueExpr});
    else
      EnumValue.getAsObject()->insert({"Value", M.Value});
    
    if (!M.Description.empty()) {
      Value ArrDesc = Array();
      for (const CommentInfo& Child : M.Description) 
        ArrDesc.getAsArray()->emplace_back(extractValue(Child));
      EnumValue.getAsObject()->insert({"EnumValueComments", ArrDesc});
    }
    Arr.getAsArray()->emplace_back(EnumValue);
  }
  Obj.insert({"EnumValues", Arr});
  
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    Obj.insert({"EnumComments", ArrDesc});
  }
  Obj.insert({"Location", extractValue(I.DefLoc, CDCtx.RepositoryUrl)});
  return Obj;
}

void extractScopeChildren(const ScopeChildren &S, Object &Obj, 
                          StringRef ParentInfoDir,
                          const ClangDocContext &CDCtx) {
  Value ArrNamespace = Array();
  for (const Reference& Child : S.Namespaces)
    ArrNamespace.getAsArray()->emplace_back(extractValue(Child, ParentInfoDir));
  
  if (!ArrNamespace.getAsArray()->empty())
    Obj.insert({"Namespace", Object{{"Links", ArrNamespace}}});
  
  Value ArrRecord = Array();
  for (const Reference& Child : S.Records)
    ArrRecord.getAsArray()->emplace_back(extractValue(Child, ParentInfoDir));
  
  if (!ArrRecord.getAsArray()->empty())
    Obj.insert({"Record", Object{{"Links", ArrRecord}}});
  
  Value ArrFunction = Array();
  Value PublicFunction = Array();
  Value ProtectedFunction = Array();
  Value PrivateFunction = Array();
  
  for (const FunctionInfo& Child : S.Functions) {
    Value F = extractValue(Child, ParentInfoDir, CDCtx);
    AccessSpecifier Access = Child.Access;
    if (Access == AccessSpecifier::AS_public)
      PublicFunction.getAsArray()->emplace_back(F);
    else if (Access == AccessSpecifier::AS_protected)
      ProtectedFunction.getAsArray()->emplace_back(F);
    else
      ArrFunction.getAsArray()->emplace_back(F);
  }  
  if (!ArrFunction.getAsArray()->empty())
    Obj.insert({"Function", Object{{"Obj", ArrFunction}}});
  
  if (!PublicFunction.getAsArray()->empty())
    Obj.insert({"PublicFunction", Object{{"Obj", PublicFunction}}});
  
  if (!ProtectedFunction.getAsArray()->empty())
    Obj.insert({"ProtectedFunction", Object{{"Obj", ProtectedFunction}}});
  
  
  Value ArrEnum = Array();
  for (const EnumInfo& Child : S.Enums)
    ArrEnum.getAsArray()->emplace_back(extractValue(Child, CDCtx));
  
  if (!ArrEnum.getAsArray()->empty())
    Obj.insert({"Enums", Object{{"Obj", ArrEnum }}});
  
  Value ArrTypedefs = Array();
  for (const TypedefInfo& Child : S.Typedefs) 
    ArrTypedefs.getAsArray()->emplace_back(extractValue(Child, CDCtx));
  
  if (!ArrTypedefs.getAsArray()->empty())
    Obj.insert({"Typedefs", Object{{"Obj", ArrTypedefs }}});
  
  
  llvm::raw_fd_ostream os(1, false);
  llvm::json::OStream jStream(os, /*Indent=*/2);
  jStream.value(ArrTypedefs);
}

Value extractValue(const NamespaceInfo &I, const ClangDocContext &CDCtx) {
  Object NamespaceValue = Object();
  std::string InfoTitle;
  if (I.Name.str() == "")
    InfoTitle = "Global Namespace";
  else
    InfoTitle = ("namespace " + I.Name).str();  
  
  StringRef BasePath = I.getRelativeFilePath("");
  NamespaceValue.insert({"NamespaceTitle", InfoTitle});
  NamespaceValue.insert({"NamespacePath", I.getRelativeFilePath("")});
  
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    NamespaceValue.insert({"NamespaceComments", ArrDesc });
  }
  extractScopeChildren(I.Children, NamespaceValue, BasePath, CDCtx);
  return NamespaceValue;
}

Value extractValue(const RecordInfo &I, const ClangDocContext &CDCtx) {
  Object RecordValue = Object();
  
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    RecordValue.insert({"RecordComments", ArrDesc });
  }
  RecordValue.insert({"Name", I.Name});
  RecordValue.insert({"FullName", extractRecordPrototype(I.FullName, 
                                                         I.Parents, 
                                                         I.VirtualParents)});
  
  RecordValue.insert({"RecordType", getTagType(I.TagType)});
  RecordValue.insert({"Location", extractValue(I.DefLoc, CDCtx.RepositoryUrl)});
  StringRef BasePath = I.getRelativeFilePath("");
  extractScopeChildren(I.Children, RecordValue, BasePath, CDCtx);
  Value PublicMembers = Array();
  Value ProtectedMembers = Array();
  Value PrivateMembers = Array();
  for (const MemberTypeInfo &Member : I.Members ) {
    Value MemberValue = Object();
    MemberValue.getAsObject()->insert({"Name", Member.Name});
    MemberValue.getAsObject()->insert({"Type", Member.Type.Name});
    if (!Member.Description.empty()) {
      Value ArrDesc = Array();
      for (const CommentInfo& Child : Member.Description) 
        ArrDesc.getAsArray()->emplace_back(extractValue(Child));
      MemberValue.getAsObject()->insert({"MemberComments", ArrDesc });
    }
    
    if (Member.Access == AccessSpecifier::AS_public)
      PublicMembers.getAsArray()->emplace_back(MemberValue);
    else if (Member.Access == AccessSpecifier::AS_protected)
      ProtectedMembers.getAsArray()->emplace_back(MemberValue);
    else if (Member.Access == AccessSpecifier::AS_private)
      PrivateMembers.getAsArray()->emplace_back(MemberValue);
  }
  if (!PublicMembers.getAsArray()->empty())
    RecordValue.insert({"PublicMembers", Object{{"Obj", PublicMembers}}});
  if (!ProtectedMembers.getAsArray()->empty())
    RecordValue.insert({"ProtectedMembers", Object{{"Obj", ProtectedMembers}}});
  if (!PrivateMembers.getAsArray()->empty())
    RecordValue.insert({"PrivateMembers", Object{{"Obj", PrivateMembers}}});
  
  return RecordValue;
}

void setupTemplateValue(const ClangDocContext &CDCtx, Value &V, Info *I) {
  V.getAsObject()->insert({"ProjectName", CDCtx.ProjectName});
  Value StylesheetArr = Array();
  auto InfoPath = I->getRelativeFilePath("");
  SmallString<128> RelativePath = computeRelativePath("", InfoPath);
  for (const auto &FilePath : CDCtx.UserStylesheets) {
    SmallString<128> StylesheetPath = RelativePath;
    llvm::sys::path::append(StylesheetPath,
                            llvm::sys::path::filename(FilePath));
    llvm::sys::path::native(StylesheetPath, llvm::sys::path::Style::posix);
    StylesheetArr.getAsArray()->emplace_back(StylesheetPath);
  }
  V.getAsObject()->insert({"Stylesheets", StylesheetArr});
  
  Value ScriptArr = Array();
  for (auto Script : CDCtx.JsScripts) {
    SmallString<128> JsPath = RelativePath;
    llvm::sys::path::append(JsPath, llvm::sys::path::filename(Script));
    ScriptArr.getAsArray()->emplace_back(JsPath);
  }
  V.getAsObject()->insert({"Scripts", ScriptArr});
}

llvm::Error
MustacheHTMLGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                          const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace: {
    Value V = extractValue(*static_cast<clang::doc::NamespaceInfo *>(I), CDCtx);
    setupTemplateValue(CDCtx, V, I);
    NamespaceTemplate->render(V, OS);
    break;
  }
  case InfoType::IT_record: {
    Value V = extractValue(*static_cast<clang::doc::RecordInfo *>(I), CDCtx);
    setupTemplateValue(CDCtx, V, I);
    // Serialize the JSON value to the output stream in a readable format.
    RecordTemplate->render(V, OS);
    break;
  }  
  case InfoType::IT_enum:
    llvm::outs() << "IT_enum\n";
    break;
  case InfoType::IT_function:
    llvm::outs() << "IT_Function\n";
    break;
  case InfoType::IT_typedef:
    llvm::outs() << "IT_typedef\n";
    break;
  case InfoType::IT_default:
    return createStringError(llvm::inconvertibleErrorCode(),
                             "unexpected InfoType");
  }
  return llvm::Error::success();
}

llvm::Error MustacheHTMLGenerator::createResources(ClangDocContext &CDCtx) {
  llvm::Error Err = llvm::Error::success();
  for (const auto &FilePath : CDCtx.UserStylesheets) {
    Err = copyFile(FilePath, CDCtx.OutDirectory);
    if (Err)
      return Err;
  }
  for (const auto &FilePath : CDCtx.JsScripts) {
    Err = copyFile(FilePath, CDCtx.OutDirectory);
    if (Err)
      return Err;
  }
  return llvm::Error::success();
}

const char *MustacheHTMLGenerator::Format = "mhtml";


static GeneratorRegistry::Add<MustacheHTMLGenerator> MHTML(MustacheHTMLGenerator::Format,
                                                           "Generator for mustache HTML output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int MHTMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang