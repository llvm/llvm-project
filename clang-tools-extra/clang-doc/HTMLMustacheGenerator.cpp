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
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

namespace clang {
namespace doc {

static std::unique_ptr<MustacheTemplateFile> NamespaceTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> RecordTemplate = nullptr;

class MustacheHTMLGenerator : public MustacheGenerator {
public:
  static const char *Format;
  Error createResources(ClangDocContext &CDCtx) override;
  Error generateDocForInfo(Info *I, raw_ostream &OS,
                           const ClangDocContext &CDCtx) override;
  Error setupTemplateFiles(const ClangDocContext &CDCtx) override;
  Error generateDocForJSON(json::Value &JSON, raw_fd_ostream &OS,
                           const ClangDocContext &CDCtx, StringRef ObjTypeStr,
                           StringRef RelativeRootPath) override;
  // Populates templates with CSS stylesheets, JS scripts paths.
  Error setupTemplateResources(const ClangDocContext &CDCtx, json::Value &V,
                               SmallString<128> RelativeRootPath);
  llvm::Error generateDocumentation(
      StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
      const ClangDocContext &CDCtx, std::string DirName) override;
};

Error MustacheHTMLGenerator::setupTemplateFiles(const ClangDocContext &CDCtx) {
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

Error MustacheHTMLGenerator::setupTemplateResources(
    const ClangDocContext &CDCtx, json::Value &V,
    SmallString<128> RelativeRootPath) {
  V.getAsObject()->insert({"ProjectName", CDCtx.ProjectName});
  json::Value StylesheetArr = Array();
  sys::path::native(RelativeRootPath, sys::path::Style::posix);

  auto *SSA = StylesheetArr.getAsArray();
  SSA->reserve(CDCtx.UserStylesheets.size());
  for (const auto &FilePath : CDCtx.UserStylesheets) {
    SmallString<128> StylesheetPath = RelativeRootPath;
    sys::path::append(StylesheetPath, sys::path::Style::posix,
                      sys::path::filename(FilePath));
    SSA->emplace_back(StylesheetPath);
  }
  V.getAsObject()->insert({"Stylesheets", StylesheetArr});

  json::Value ScriptArr = Array();
  auto *SCA = ScriptArr.getAsArray();
  SCA->reserve(CDCtx.JsScripts.size());
  for (auto Script : CDCtx.JsScripts) {
    SmallString<128> JsPath = RelativeRootPath;
    sys::path::append(JsPath, sys::path::Style::posix,
                      sys::path::filename(Script));
    SCA->emplace_back(JsPath);
  }
  V.getAsObject()->insert({"Scripts", ScriptArr});
  return Error::success();
}

Error MustacheHTMLGenerator::generateDocForJSON(json::Value &JSON,
                                                raw_fd_ostream &OS,
                                                const ClangDocContext &CDCtx,
                                                StringRef ObjTypeStr,
                                                StringRef RelativeRootPath) {
  if (ObjTypeStr == "namespace") {
    if (auto Err = setupTemplateResources(CDCtx, JSON, RelativeRootPath))
      return Err;
    assert(NamespaceTemplate && "NamespaceTemplate is nullptr.");
    NamespaceTemplate->render(JSON, OS);
  } else if (ObjTypeStr == "record") {
    if (auto Err = setupTemplateResources(CDCtx, JSON, RelativeRootPath))
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

Error MustacheHTMLGenerator::generateDocumentation(
    StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
    const ClangDocContext &CDCtx, std::string DirName) {
  return MustacheGenerator::generateDocumentation(RootDir, std::move(Infos),
                                                  CDCtx, "html");
}

const char *MustacheHTMLGenerator::Format = "mustache";

static GeneratorRegistry::Add<MustacheHTMLGenerator>
    MHTML(MustacheHTMLGenerator::Format, "Generator for mustache HTML output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int MHTMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
