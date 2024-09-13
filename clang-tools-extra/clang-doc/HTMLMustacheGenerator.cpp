//===-- HTMLMustacheGenerator.cpp - HTML Mustache Generator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Generators.h"
#include "Representation.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mustache.h"

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

namespace clang {
namespace doc {


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
    if (auto EC = BufferOrError.getError()) {
      return llvm::createFileError("cannot open file", EC);
    }
    std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(BufferOrError.get());
    llvm::StringRef FileContent = Buffer->getBuffer();
    registerPartial(Name, FileContent);
    return llvm::Error::success();
  }

  MustacheTemplateFile(StringRef TemplateStr) : Template(TemplateStr) {}
};

static std::unique_ptr<MustacheTemplateFile> NamespaceTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> RecordTemplate = nullptr;


llvm::Error 
setupTemplateFiles(const clang::doc::ClangDocContext &CDCtx) {
  auto TemplateFilePath = CDCtx.MustacheTemplates.lookup("template");
  auto Template = MustacheTemplateFile::createMustacheFile(TemplateFilePath);
  if (auto EC = Template.getError()) {
    return llvm::createFileError("cannot open file", EC);
  }
  NamespaceTemplate = std::move(Template.get());
  return llvm::Error::success();
}

llvm::Error 
MustacheHTMLGenerator::generateDocs(llvm::StringRef RootDir, 
                                    llvm::StringMap<std::unique_ptr<doc::Info>> Infos, 
                                    const clang::doc::ClangDocContext &CDCtx) {
  if (auto Err = setupTemplateFiles(CDCtx)) {
    return Err;
  }
  // Track which directories we already tried to create.
  llvm::StringSet<> CreatedDirs;
  // Collect all output by file name and create the necessary directories.
  llvm::StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    doc::Info *Info = Group.getValue().get();
    
    llvm::SmallString<128> Path;
    llvm::sys::path::native(RootDir, Path);
    llvm::sys::path::append(Path, Info->getRelativeFilePath(""));
    if (!CreatedDirs.contains(Path)) {
      if (std::error_code Err = llvm::sys::fs::create_directories(Path);
          Err != std::error_code()) {
        return llvm::createStringError(Err, "Failed to create directory '%s'.",
                                       Path.c_str());
      }
      CreatedDirs.insert(Path);
    }

    llvm::sys::path::append(Path, Info->getFileBaseName() + ".html");
    FileToInfos[Path].push_back(Info);
  }
  
  for (const auto &Group : FileToInfos) {
    std::error_code FileErr;
    llvm::raw_fd_ostream InfoOS(Group.getKey(), FileErr,
                                llvm::sys::fs::OF_None);
    if (FileErr) {
      return llvm::createStringError(FileErr, "Error opening file '%s'",
                                     Group.getKey().str().c_str());
    }
    for (const auto &Info : Group.getValue()) {
      if (llvm::Error Err = generateDocForInfo(Info, InfoOS, CDCtx)) {
        return Err;
      }
    }
  }
  return llvm::Error::success();
}

Value extractValue(const Location &L, 
                   std::optional<StringRef> RepositoryUrl = std::nullopt) {
  Object Obj = Object();
  if (!L.IsFileInRootDir || !RepositoryUrl) {
    Obj.insert({"LineNumber", L.LineNumber});
    Obj.insert({"Filename", L.Filename});
  }
  SmallString<128> FileURL(*RepositoryUrl);
  llvm::sys::path::append(FileURL, llvm::sys::path::Style::posix, L.Filename);
  Obj.insert({"FileURL", FileURL});
  
  return Obj;
}

Value extractValue(const Reference &I, StringRef CurrentDirectory) {
  llvm::SmallString<64> Path = I.getRelativeFilePath(CurrentDirectory);
  llvm::sys::path::append(Path, I.getFileBaseName() + ".html");
  llvm::sys::path::native(Path, llvm::sys::path::Style::posix);
  Object Obj = Object();
  Obj.insert({"Link", Path});
  Obj.insert({"Name", I.Name});
  return Obj;
}


Value extractValue(const TypedefInfo &I) {
  // Not Supported
  return nullptr;
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

Value extractValue(const FunctionInfo &I, StringRef ParentInfoDir,
                   const ClangDocContext &CDCtx) {
  Object Obj = Object();
  Obj.insert({"Name", I.Name});
  Obj.insert({"ID", llvm::toHex(llvm::toStringRef(I.USR))});
  Obj.insert({"Access", getAccessSpelling(I.Access).str()});
  Obj.insert({"ReturnType", extractValue(I.ReturnType.Type, ParentInfoDir)});
  
  Value ParamArr = Array();
  for (const auto &P : I.Params) {
    ParamArr.getAsArray()->emplace_back(extractValue(P.Type, ParentInfoDir));
  }
  Obj.insert({"Params", ParamArr});
  
  if (!I.Description.empty()) {
    Value ArrDesc = Array();
    for (const CommentInfo& Child : I.Description) 
      ArrDesc.getAsArray()->emplace_back(extractValue(Child));
    Obj.insert({"FunctionComments", ArrDesc});
  }
  if (I.DefLoc) {
    if (!CDCtx.RepositoryUrl)
      Obj.insert({"Location", extractValue(*I.DefLoc)});
    else
      Obj.insert({"Location", extractValue(*I.DefLoc, 
                                           StringRef{*CDCtx.RepositoryUrl})});  
  }
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
  
  if (I.DefLoc) {
    if (!CDCtx.RepositoryUrl)
      Obj.insert({"Location", extractValue(*I.DefLoc)});
    else
      Obj.insert({"Location", extractValue(*I.DefLoc, 
                                           StringRef{*CDCtx.RepositoryUrl})});  
  }
  
  return Obj;
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

  Value ArrNamespace = Array();
  for (const Reference& Child : I.Children.Namespaces)
    ArrNamespace.getAsArray()->emplace_back(extractValue(Child, BasePath));
  
  if (!ArrNamespace.getAsArray()->empty())
    NamespaceValue.insert({"Namespace", Object{{"Links", ArrNamespace}}});
  
  Value ArrRecord = Array();
  for (const Reference& Child : I.Children.Records)
    ArrRecord.getAsArray()->emplace_back(extractValue(Child, BasePath));
  
  if (!ArrRecord.getAsArray()->empty())
    NamespaceValue.insert({"Record", Object{{"Links", ArrRecord}}});
  
  Value ArrFunction = Array();
  for (const FunctionInfo& Child : I.Children.Functions)
    ArrFunction.getAsArray()->emplace_back(extractValue(Child, BasePath,
                                                        CDCtx));
  if (!ArrFunction.getAsArray()->empty())
    NamespaceValue.insert({"Function", Object{{"Obj", ArrFunction}}});
  
  Value ArrEnum = Array();
  for (const EnumInfo& Child : I.Children.Enums)
    ArrEnum.getAsArray()->emplace_back(extractValue(Child, CDCtx));
  
  if (!ArrEnum.getAsArray()->empty())
    NamespaceValue.insert({"Enums", Object{{"Obj", ArrEnum }}});
  
  Value ArrTypedefs = Array();
  for (const TypedefInfo& Child : I.Children.Typedefs) 
    ArrTypedefs.getAsArray()->emplace_back(extractValue(Child));
  
  if (!ArrTypedefs.getAsArray()->empty())
    NamespaceValue.insert({"Typedefs", Object{{"Obj", ArrTypedefs }}});
  
  return NamespaceValue;
}



llvm::Error
MustacheHTMLGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                          const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace: {
    Value V = extractValue(*static_cast<clang::doc::NamespaceInfo *>(I), CDCtx);
    llvm::outs() << V << "\n";
    OS << NamespaceTemplate->render(V);
    break;
  }
  case InfoType::IT_record:
    break;
  case InfoType::IT_enum:
    break;
  case InfoType::IT_function:
    break;
  case InfoType::IT_typedef:
    break;
  case InfoType::IT_default:
    return createStringError(llvm::inconvertibleErrorCode(),
                             "unexpected InfoType");
  }
  return llvm::Error::success();
}

llvm::Error MustacheHTMLGenerator::createResources(ClangDocContext &CDCtx) {
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