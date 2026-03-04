//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains the Markdown generator using Mustache template files.
///
//===----------------------------------------------------------------------===//

#include "Generators.h"

namespace clang {
using namespace llvm;
namespace doc {
static std::unique_ptr<MustacheTemplateFile> RecordTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> NamespaceTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> AllFilesTemplate = nullptr;

static std::unique_ptr<MustacheTemplateFile> IndexTemplate = nullptr;

struct MDMustacheGenerator : public MustacheGenerator {
  static const char *Format;
  Error generateDocumentation(StringRef RootDir,
                              StringMap<std::unique_ptr<doc::Info>> Infos,
                              const ClangDocContext &CDCtx,
                              std::string DirName) override;
  Error setupTemplateFiles(const ClangDocContext &CDCtx) override;
  Error generateDocForJSON(json::Value &JSON, raw_fd_ostream &OS,
                           const ClangDocContext &CDCtx,
                           StringRef ObjectTypeStr,
                           StringRef RelativeRootPath) override;
  // This generator doesn't need this function, but it inherits from the
  // original generator interface.
  Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                           const ClangDocContext &CDCtx) override;
};

Error MDMustacheGenerator::setupTemplateFiles(const ClangDocContext &CDCtx) {
  std::string ClassFilePath = CDCtx.MustacheTemplates.lookup("class-template");
  std::string NamespaceFilePath =
      CDCtx.MustacheTemplates.lookup("namespace-template");
  std::string AllFilesPath = CDCtx.MustacheTemplates.lookup("all-files");
  std::string IndexFilePath = CDCtx.MustacheTemplates.lookup("index");
  std::string CommentsFilePath = CDCtx.MustacheTemplates.lookup("comments");
  std::vector<std::pair<StringRef, StringRef>> Partials = {
      {"Comments", CommentsFilePath}};

  if (Error Err = setupTemplate(RecordTemplate, ClassFilePath, Partials))
    return Err;
  if (Error Err = setupTemplate(NamespaceTemplate, NamespaceFilePath, Partials))
    return Err;
  if (Error Err = setupTemplate(AllFilesTemplate, AllFilesPath, Partials))
    return Err;
  if (Error Err = setupTemplate(IndexTemplate, IndexFilePath, Partials))
    return Err;

  // Override the default HTML Mustache escape characters. We don't need to
  // override `<` here.
  static const DenseMap<char, std::string> EscapeChars;
  RecordTemplate->setEscapeCharacters(EscapeChars);
  NamespaceTemplate->setEscapeCharacters(EscapeChars);
  AllFilesTemplate->setEscapeCharacters(EscapeChars);
  IndexTemplate->setEscapeCharacters(EscapeChars);

  return Error::success();
}

Error MDMustacheGenerator::generateDocumentation(
    StringRef RootDir, StringMap<std::unique_ptr<doc::Info>> Infos,
    const clang::doc::ClangDocContext &CDCtx, std::string Dirname) {
  return MustacheGenerator::generateDocumentation(RootDir, std::move(Infos),
                                                  CDCtx, "md");
}

Error MDMustacheGenerator::generateDocForJSON(json::Value &JSON,
                                              raw_fd_ostream &OS,
                                              const ClangDocContext &CDCtx,
                                              StringRef ObjTypeStr,
                                              StringRef RelativeRootPath) {
  if (ObjTypeStr == "record") {
    assert(RecordTemplate && "RecordTemplate is nullptr.");
    RecordTemplate->render(JSON, OS);
  } else if (ObjTypeStr == "namespace") {
    assert(NamespaceTemplate && "NamespaceTemplate is nullptr.");
    NamespaceTemplate->render(JSON, OS);
  } else if (ObjTypeStr == "all_files") {
    assert(AllFilesTemplate && "AllFilesTemplate is nullptr.");
    AllFilesTemplate->render(JSON, OS);
  } else if (ObjTypeStr == "index") {
    assert(IndexTemplate && "IndexTemplate is nullptr");
    IndexTemplate->render(JSON, OS);
  }
  return Error::success();
}

Error MDMustacheGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                              const ClangDocContext &CDCtx) {
  return Error::success();
}

const char *MDMustacheGenerator::Format = "md_mustache";

static GeneratorRegistry::Add<MDMustacheGenerator>
    MDMustache(MDMustacheGenerator::Format,
               "Generator for mustache Markdown output.");

volatile int MDMustacheGeneratorAnchorSource = 0;
} // namespace doc
} // namespace clang
