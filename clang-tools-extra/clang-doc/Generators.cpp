//===-- Generators.cpp - Generator Registry ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "support/File.h"
#include "llvm/Support/TimeProfiler.h"

LLVM_INSTANTIATE_REGISTRY(clang::doc::GeneratorRegistry)

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

namespace clang {
namespace doc {

llvm::Expected<std::unique_ptr<Generator>>
findGeneratorByName(llvm::StringRef Format) {
  for (const auto &Generator : GeneratorRegistry::entries()) {
    if (Generator.getName() != Format)
      continue;
    return Generator.instantiate();
  }
  return createStringError(llvm::inconvertibleErrorCode(),
                           "can't find generator: " + Format);
}

// Enum conversion

std::string getTagType(TagTypeKind AS) {
  switch (AS) {
  case TagTypeKind::Class:
    return "class";
  case TagTypeKind::Union:
    return "union";
  case TagTypeKind::Interface:
    return "interface";
  case TagTypeKind::Struct:
    return "struct";
  case TagTypeKind::Enum:
    return "enum";
  }
  llvm_unreachable("Unknown TagTypeKind");
}

Error createFileOpenError(StringRef FileName, std::error_code EC) {
  return createFileError("cannot open file " + FileName, EC);
}

Error MustacheGenerator::setupTemplate(
    std::unique_ptr<MustacheTemplateFile> &Template, StringRef TemplatePath,
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

Error MustacheGenerator::generateDocumentation(
    StringRef RootDir, StringMap<std::unique_ptr<doc::Info>> Infos,
    const clang::doc::ClangDocContext &CDCtx, std::string DirName) {
  {
    llvm::TimeTraceScope TS("Setup Templates");
    if (auto Err = setupTemplateFiles(CDCtx))
      return Err;
  }

  {
    llvm::TimeTraceScope TS("Generate JSON for Mustache");
    if (auto JSONGenerator = findGeneratorByName("json")) {
      if (Error Err = JSONGenerator.get()->generateDocumentation(
              RootDir, std::move(Infos), CDCtx))
        return Err;
    } else
      return JSONGenerator.takeError();
  }

  SmallString<128> JSONPath;
  sys::path::native(RootDir.str() + "/json", JSONPath);

  {
    llvm::TimeTraceScope TS("Iterate JSON files");
    std::error_code EC;
    sys::fs::recursive_directory_iterator JSONIter(JSONPath, EC);
    std::vector<json::Value> JSONFiles;
    JSONFiles.reserve(Infos.size());
    if (EC)
      return createStringError("Failed to create directory iterator.");

    SmallString<128> DocsDirPath(RootDir.str() + '/' + DirName);
    sys::path::native(DocsDirPath);
    if (auto EC = sys::fs::create_directories(DocsDirPath))
      return createFileError(DocsDirPath, EC);
    while (JSONIter != sys::fs::recursive_directory_iterator()) {
      // create the same directory structure in the docs format dir
      if (JSONIter->type() == sys::fs::file_type::directory_file) {
        SmallString<128> DocsClonedPath(JSONIter->path());
        sys::path::replace_path_prefix(DocsClonedPath, JSONPath, DocsDirPath);
        if (auto EC = sys::fs::create_directories(DocsClonedPath)) {
          return createFileError(DocsClonedPath, EC);
        }
      }

      if (EC)
        return createFileError("Failed to iterate: " + JSONIter->path(), EC);

      auto Path = StringRef(JSONIter->path());
      if (!Path.ends_with(".json")) {
        JSONIter.increment(EC);
        continue;
      }

      auto File = MemoryBuffer::getFile(Path);
      if (EC = File.getError(); EC) {
        // TODO: Buffer errors to report later, look into using Clang
        // diagnostics.
        llvm::errs() << "Failed to open file: " << Path << " " << EC.message()
                     << '\n';
      }

      auto Parsed = json::parse((*File)->getBuffer());
      if (!Parsed)
        return Parsed.takeError();
      auto ValidJSON = Parsed.get();

      std::error_code FileErr;
      SmallString<128> DocsFilePath(JSONIter->path());
      sys::path::replace_path_prefix(DocsFilePath, JSONPath, DocsDirPath);
      sys::path::replace_extension(DocsFilePath, DirName);
      raw_fd_ostream InfoOS(DocsFilePath, FileErr, sys::fs::OF_None);
      if (FileErr)
        return createFileOpenError(Path, FileErr);

      auto RelativeRootPath = getRelativePathToRoot(DocsFilePath, DocsDirPath);
      auto InfoTypeStr =
          getInfoTypeStr(Parsed->getAsObject(), sys::path::stem(DocsFilePath));
      if (!InfoTypeStr)
        return InfoTypeStr.takeError();
      if (Error Err = generateDocForJSON(*Parsed, InfoOS, CDCtx,
                                         InfoTypeStr.get(), RelativeRootPath))
        return Err;
      JSONIter.increment(EC);
    }
  }

  return Error::success();
}

Expected<std::string> MustacheGenerator::getInfoTypeStr(Object *Info,
                                                        StringRef Filename) {
  auto StrValue = (*Info)["InfoType"];
  if (StrValue.kind() != json::Value::Kind::String)
    return createStringError("JSON file '%s' does not contain key: 'InfoType'.",
                             Filename.str().c_str());
  auto ObjTypeStr = StrValue.getAsString();
  if (!ObjTypeStr.has_value())
    return createStringError(
        "JSON file '%s' does not contain 'InfoType' field as a string.",
        Filename.str().c_str());
  return ObjTypeStr.value().str();
}

SmallString<128>
MustacheGenerator::getRelativePathToRoot(StringRef PathToFile,
                                         StringRef DocsRootPath) {
  SmallString<128> PathVec(PathToFile);
  // Remove filename, or else the relative path will have an extra "../"
  sys::path::remove_filename(PathVec);
  return computeRelativePath(DocsRootPath, PathVec);
}

llvm::Error Generator::createResources(ClangDocContext &CDCtx) {
  return llvm::Error::success();
}

// A function to add a reference to Info in Idx.
// Given an Info X with the following namespaces: [B,A]; a reference to X will
// be added in the children of a reference to B, which should be also a child of
// a reference to A, where A is a child of Idx.
//   Idx
//    |-- A
//        |--B
//           |--X
// If the references to the namespaces do not exist, they will be created. If
// the references already exist, the same one will be used.
void Generator::addInfoToIndex(Index &Idx, const doc::Info *Info) {
  // Index pointer that will be moving through Idx until the first parent
  // namespace of Info (where the reference has to be inserted) is found.
  Index *I = &Idx;
  // The Namespace vector includes the upper-most namespace at the end so the
  // loop will start from the end to find each of the namespaces.
  for (const auto &R : llvm::reverse(Info->Namespace)) {
    // Look for the current namespace in the children of the index I is
    // pointing.
    auto It = llvm::find(I->Children, R.USR);
    if (It != I->Children.end()) {
      // If it is found, just change I to point the namespace reference found.
      I = &*It;
    } else {
      // If it is not found a new reference is created
      I->Children.emplace_back(R.USR, R.Name, R.RefType, R.Path);
      // I is updated with the reference of the new namespace reference
      I = &I->Children.back();
    }
  }
  // Look for Info in the vector where it is supposed to be; it could already
  // exist if it is a parent namespace of an Info already passed to this
  // function.
  auto It = llvm::find(I->Children, Info->USR);
  if (It == I->Children.end()) {
    // If it is not in the vector it is inserted
    I->Children.emplace_back(Info->USR, Info->extractName(), Info->IT,
                             Info->Path);
  } else {
    // If it not in the vector we only check if Path and Name are not empty
    // because if the Info was included by a namespace it may not have those
    // values.
    if (It->Path.empty())
      It->Path = Info->Path;
    if (It->Name.empty())
      It->Name = Info->extractName();
  }
}

// This anchor is used to force the linker to link in the generated object file
// and thus register the generators.
[[maybe_unused]] static int YAMLGeneratorAnchorDest = YAMLGeneratorAnchorSource;
[[maybe_unused]] static int MDGeneratorAnchorDest = MDGeneratorAnchorSource;
[[maybe_unused]] static int HTMLGeneratorAnchorDest = HTMLGeneratorAnchorSource;
[[maybe_unused]] static int MHTMLGeneratorAnchorDest =
    MHTMLGeneratorAnchorSource;
[[maybe_unused]] static int JSONGeneratorAnchorDest = JSONGeneratorAnchorSource;
} // namespace doc
} // namespace clang
