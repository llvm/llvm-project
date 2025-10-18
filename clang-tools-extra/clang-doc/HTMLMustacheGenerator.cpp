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
static Error generateDocForJSON(json::Value &JSON, StringRef Filename,
                                StringRef Path, raw_fd_ostream &OS,
                                const ClangDocContext &CDCtx);

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

class MustacheTemplateFile {
  BumpPtrAllocator Allocator;
  StringSaver Saver;
  MustacheContext Ctx;
  Template T;
  std::unique_ptr<MemoryBuffer> Buffer;

public:
  static Expected<std::unique_ptr<MustacheTemplateFile>>
  createMustacheFile(StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);
    return std::make_unique<MustacheTemplateFile>(
        std::move(BufferOrError.get()));
  }

  Error registerPartialFile(StringRef Name, StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
        MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);

    std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrError.get());
    StringRef FileContent = Buffer->getBuffer();
    T.registerPartial(Name.str(), FileContent.str());
    return Error::success();
  }

  void render(json::Value &V, raw_ostream &OS) { T.render(V, OS); }

  MustacheTemplateFile(std::unique_ptr<MemoryBuffer> &&B)
      : Saver(Allocator), Ctx(Allocator, Saver), T(B->getBuffer(), Ctx),
        Buffer(std::move(B)) {}
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

  {
    llvm::TimeTraceScope TS("Generate JSON for Mustache");
    if (auto JSONGenerator = findGeneratorByName("json")) {
      if (Error Err = JSONGenerator.get()->generateDocs(
              RootDir, std::move(Infos), CDCtx))
        return Err;
    } else
      return JSONGenerator.takeError();
  }
  SmallString<128> JSONPath;
  sys::path::native(RootDir.str() + "/json", JSONPath);

  StringMap<json::Value> JSONFileMap;
  {
    llvm::TimeTraceScope TS("Iterate JSON files");
    std::error_code EC;
    sys::fs::directory_iterator JSONIter(JSONPath, EC);
    std::vector<json::Value> JSONFiles;
    JSONFiles.reserve(Infos.size());
    if (EC)
      return createStringError("Failed to create directory iterator.");

    SmallString<128> HTMLDirPath(RootDir.str() + "/html/");
    if (auto EC = sys::fs::create_directories(HTMLDirPath))
      return createFileError(HTMLDirPath, EC);
    while (JSONIter != sys::fs::directory_iterator()) {
      if (EC)
        return createFileError("Failed to iterate: " + JSONIter->path(), EC);

      auto Path = StringRef(JSONIter->path());
      if (!Path.ends_with(".json")) {
        JSONIter.increment(EC);
        continue;
      }

      auto File = MemoryBuffer::getFile(Path);
      if (EC = File.getError(); EC)
        // TODO: Buffer errors to report later, look into using Clang
        // diagnostics.
        llvm::errs() << "Failed to open file: " << Path << " " << EC.message()
                     << '\n';

      auto Parsed = json::parse((*File)->getBuffer());
      if (!Parsed)
        return Parsed.takeError();

      std::error_code FileErr;
      SmallString<128> HTMLFilePath(HTMLDirPath);
      sys::path::append(HTMLFilePath, sys::path::filename(Path));
      sys::path::replace_extension(HTMLFilePath, "html");
      raw_fd_ostream InfoOS(HTMLFilePath, FileErr, sys::fs::OF_None);
      if (FileErr)
        return createFileOpenError(Path, FileErr);

      if (Error Err = generateDocForJSON(*Parsed, sys::path::stem(HTMLFilePath),
                                         HTMLFilePath, InfoOS, CDCtx))
        return Err;
      JSONIter.increment(EC);
    }
  }

  return Error::success();
}

static Error setupTemplateValue(const ClangDocContext &CDCtx, json::Value &V) {
  V.getAsObject()->insert({"ProjectName", CDCtx.ProjectName});
  json::Value StylesheetArr = Array();
  SmallString<128> RelativePath("./");
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

static Error generateDocForJSON(json::Value &JSON, StringRef Filename,
                                StringRef Path, raw_fd_ostream &OS,
                                const ClangDocContext &CDCtx) {
  auto StrValue = (*JSON.getAsObject())["InfoType"];
  if (StrValue.kind() != json::Value::Kind::String)
    return createStringError("JSON file '%s' does not contain key: 'InfoType'.",
                             Filename.str().c_str());
  auto ObjTypeStr = StrValue.getAsString();
  if (!ObjTypeStr.has_value())
    return createStringError(
        "JSON file '%s' does not contain 'InfoType' field as a string.",
        Filename.str().c_str());

  if (ObjTypeStr.value() == "namespace") {
    if (auto Err = setupTemplateValue(CDCtx, JSON))
      return Err;
    assert(NamespaceTemplate && "NamespaceTemplate is nullptr.");
    NamespaceTemplate->render(JSON, OS);
  } else if (ObjTypeStr.value() == "record") {
    if (auto Err = setupTemplateValue(CDCtx, JSON))
      return Err;
    assert(RecordTemplate && "RecordTemplate is nullptr.");
    RecordTemplate->render(JSON, OS);
  }
  return Error::success();
}

Error MustacheHTMLGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                                const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_enum:
  case InfoType::IT_function:
  case InfoType::IT_typedef:
  case InfoType::IT_namespace:
  case InfoType::IT_record:
  case InfoType::IT_concept:
  case InfoType::IT_variable:
  case InfoType::IT_friend:
    break;
  case InfoType::IT_default:
    return createStringError(inconvertibleErrorCode(), "unexpected InfoType");
  }
  return Error::success();
}

Error MustacheHTMLGenerator::createResources(ClangDocContext &CDCtx) {
  std::string ResourcePath(CDCtx.OutDirectory + "/html");
  for (const auto &FilePath : CDCtx.UserStylesheets)
    if (Error Err = copyFile(FilePath, ResourcePath))
      return Err;
  for (const auto &FilePath : CDCtx.JsScripts)
    if (Error Err = copyFile(FilePath, ResourcePath))
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
