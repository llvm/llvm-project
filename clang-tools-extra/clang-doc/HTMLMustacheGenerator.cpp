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
  if (auto Err = setupTemplateFiles(CDCtx))
    return Err;
  // Track which directories we already tried to create.
  StringSet<> CreatedDirs;
  // Collect all output by file name and create the necessary directories.
  StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
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

  for (const auto &Group : FileToInfos) {
    std::error_code FileErr;
    raw_fd_ostream InfoOS(Group.getKey(), FileErr, sys::fs::OF_None);
    if (FileErr)
      return createFileOpenError(Group.getKey(), FileErr);

    for (const auto &Info : Group.getValue())
      if (Error Err = generateDocForInfo(Info, InfoOS, CDCtx))
        return Err;
  }
  return Error::success();
}

static json::Value extractValue(const NamespaceInfo &I,
                                const ClangDocContext &CDCtx) {
  Object NamespaceValue = Object();
  return NamespaceValue;
}

static json::Value extractValue(const RecordInfo &I,
                                const ClangDocContext &CDCtx) {
  Object RecordValue = Object();
  return RecordValue;
}

static Error setupTemplateValue(const ClangDocContext &CDCtx, json::Value &V,
                                Info *I) {
  return createStringError(inconvertibleErrorCode(),
                           "setupTemplateValue is unimplemented");
}

Error MustacheHTMLGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                                const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace: {
    json::Value V =
        extractValue(*static_cast<clang::doc::NamespaceInfo *>(I), CDCtx);
    if (auto Err = setupTemplateValue(CDCtx, V, I))
      return Err;
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
