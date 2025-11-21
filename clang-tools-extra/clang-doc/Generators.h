//===-- Generators.h - ClangDoc Generator ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Generator classes for converting declaration information into documentation
// in a specified format.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H

#include "Representation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Mustache.h"
#include "llvm/Support/Registry.h"

namespace clang {
namespace doc {

// Abstract base class for generators.
// This is expected to be implemented and exposed via the GeneratorRegistry.
class Generator {
public:
  virtual ~Generator() = default;

  // Write out the decl info for the objects in the given map in the specified
  // format.
  virtual llvm::Error generateDocumentation(
      StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
      const ClangDocContext &CDCtx, std::string DirName = "") = 0;

  // This function writes a file with the index previously constructed.
  // It can be overwritten by any of the inherited generators.
  // If the override method wants to run this it should call
  // Generator::createResources(CDCtx);
  virtual llvm::Error createResources(ClangDocContext &CDCtx);

  // Write out one specific decl info to the destination stream.
  virtual llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                         const ClangDocContext &CDCtx) = 0;

  static void addInfoToIndex(Index &Idx, const doc::Info *Info);
};

typedef llvm::Registry<Generator> GeneratorRegistry;

llvm::Expected<std::unique_ptr<Generator>>
findGeneratorByName(llvm::StringRef Format);

std::string getTagType(TagTypeKind AS);

llvm::Error createFileOpenError(StringRef FileName, std::error_code EC);

class MustacheTemplateFile {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver;
  llvm::mustache::MustacheContext Ctx;
  llvm::mustache::Template T;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

public:
  static Expected<std::unique_ptr<MustacheTemplateFile>>
  createMustacheFile(StringRef FileName) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrError =
        llvm::MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);
    return std::make_unique<MustacheTemplateFile>(
        std::move(BufferOrError.get()));
  }

  llvm::Error registerPartialFile(StringRef Name, StringRef FileName) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrError =
        llvm::MemoryBuffer::getFile(FileName);
    if (auto EC = BufferOrError.getError())
      return createFileOpenError(FileName, EC);

    std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(BufferOrError.get());
    StringRef FileContent = Buffer->getBuffer();
    T.registerPartial(Name.str(), FileContent.str());
    return llvm::Error::success();
  }

  void render(llvm::json::Value &V, raw_ostream &OS) { T.render(V, OS); }

  MustacheTemplateFile(std::unique_ptr<llvm::MemoryBuffer> &&B)
      : Saver(Allocator), Ctx(Allocator, Saver), T(B->getBuffer(), Ctx),
        Buffer(std::move(B)) {}
};

struct MustacheGenerator : public Generator {
  Expected<std::string> getInfoTypeStr(llvm::json::Object *Info,
                                       StringRef Filename);

  /// Used to find the relative path from the file to the format's docs root.
  /// Mainly used for the HTML resource paths.
  SmallString<128> getRelativePathToRoot(StringRef PathToFile,
                                         StringRef DocsRootPath);
  virtual ~MustacheGenerator() = default;

  /// Initializes the template files from disk and calls setupTemplate to
  /// register partials
  virtual llvm::Error setupTemplateFiles(const ClangDocContext &CDCtx) = 0;

  /// Populates templates with data from JSON and calls any specifics for the
  /// format. For example, for HTML it will render the paths for CSS and JS.
  virtual llvm::Error generateDocForJSON(llvm::json::Value &JSON,
                                         llvm::raw_fd_ostream &OS,
                                         const ClangDocContext &CDCtx,
                                         StringRef ObjectTypeStr,
                                         StringRef RelativeRootPath) = 0;

  /// Registers partials to templates.
  llvm::Error
  setupTemplate(std::unique_ptr<MustacheTemplateFile> &Template,
                StringRef TemplatePath,
                std::vector<std::pair<StringRef, StringRef>> Partials);

  /// \brief The main orchestrator for Mustache-based documentation.
  ///
  /// 1. Initializes templates files from disk by calling setupTemplateFiles.
  /// 2. Calls the JSON generator to write JSON to disk.
  /// 3. Iterates over the JSON files, recreates the directory structure from
  /// JSON, and calls generateDocForJSON for each file.
  /// 4. A file of the desired format is created.
  llvm::Error generateDocumentation(
      StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
      const clang::doc::ClangDocContext &CDCtx, std::string DirName) override;
};

// This anchor is used to force the linker to link in the generated object file
// and thus register the generators.
extern volatile int YAMLGeneratorAnchorSource;
extern volatile int MDGeneratorAnchorSource;
extern volatile int HTMLGeneratorAnchorSource;
extern volatile int MHTMLGeneratorAnchorSource;
extern volatile int JSONGeneratorAnchorSource;

} // namespace doc
} // namespace clang

namespace llvm {
extern template class Registry<clang::doc::Generator>;
} // namespace llvm

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H
