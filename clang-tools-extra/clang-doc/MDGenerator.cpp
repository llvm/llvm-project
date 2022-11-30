//===-- MDGenerator.cpp - Markdown Generator --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "Representation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace llvm;

namespace clang {
namespace doc {

// Markdown generation

static std::string genItalic(const Twine &Text) {
  return "*" + Text.str() + "*";
}

static std::string genEmphasis(const Twine &Text) {
  return "**" + Text.str() + "**";
}

static std::string
genReferenceList(const llvm::SmallVectorImpl<Reference> &Refs) {
  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);
  for (const auto &R : Refs) {
    if (&R != Refs.begin())
      Stream << ", ";
    Stream << R.Name;
  }
  return Stream.str();
}

static void writeLine(const Twine &Text, raw_ostream &OS) {
  OS << Text << "\n\n";
}

static void writeNewLine(raw_ostream &OS) { OS << "\n\n"; }

static void writeHeader(const Twine &Text, unsigned int Num, raw_ostream &OS) {
  OS << std::string(Num, '#') + " " + Text << "\n\n";
}

static void writeFileDefinition(const ClangDocContext &CDCtx, const Location &L,
                                raw_ostream &OS) {

  if (!CDCtx.RepositoryUrl) {
    OS << "*Defined at " << L.Filename << "#" << std::to_string(L.LineNumber)
       << "*";
  } else {
    OS << "*Defined at [" << L.Filename << "#" << std::to_string(L.LineNumber)
       << "](" << StringRef{*CDCtx.RepositoryUrl}
       << llvm::sys::path::relative_path(L.Filename) << "#"
       << std::to_string(L.LineNumber) << ")"
       << "*";
  }
  OS << "\n\n";
}

static void writeDescription(const CommentInfo &I, raw_ostream &OS) {
  if (I.Kind == "FullComment") {
    for (const auto &Child : I.Children)
      writeDescription(*Child, OS);
  } else if (I.Kind == "ParagraphComment") {
    for (const auto &Child : I.Children)
      writeDescription(*Child, OS);
    writeNewLine(OS);
  } else if (I.Kind == "BlockCommandComment") {
    OS << genEmphasis(I.Name);
    for (const auto &Child : I.Children)
      writeDescription(*Child, OS);
  } else if (I.Kind == "InlineCommandComment") {
    OS << genEmphasis(I.Name) << " " << I.Text;
  } else if (I.Kind == "ParamCommandComment") {
    std::string Direction = I.Explicit ? (" " + I.Direction).str() : "";
    OS << genEmphasis(I.ParamName) << I.Text << Direction << "\n\n";
  } else if (I.Kind == "TParamCommandComment") {
    std::string Direction = I.Explicit ? (" " + I.Direction).str() : "";
    OS << genEmphasis(I.ParamName) << I.Text << Direction << "\n\n";
  } else if (I.Kind == "VerbatimBlockComment") {
    for (const auto &Child : I.Children)
      writeDescription(*Child, OS);
  } else if (I.Kind == "VerbatimBlockLineComment") {
    OS << I.Text;
    writeNewLine(OS);
  } else if (I.Kind == "VerbatimLineComment") {
    OS << I.Text;
    writeNewLine(OS);
  } else if (I.Kind == "HTMLStartTagComment") {
    if (I.AttrKeys.size() != I.AttrValues.size())
      return;
    std::string Buffer;
    llvm::raw_string_ostream Attrs(Buffer);
    for (unsigned Idx = 0; Idx < I.AttrKeys.size(); ++Idx)
      Attrs << " \"" << I.AttrKeys[Idx] << "=" << I.AttrValues[Idx] << "\"";

    std::string CloseTag = I.SelfClosing ? "/>" : ">";
    writeLine("<" + I.Name + Attrs.str() + CloseTag, OS);
  } else if (I.Kind == "HTMLEndTagComment") {
    writeLine("</" + I.Name + ">", OS);
  } else if (I.Kind == "TextComment") {
    OS << I.Text;
  } else {
    OS << "Unknown comment kind: " << I.Kind << ".\n\n";
  }
}

static void writeNameLink(const StringRef &CurrentPath, const Reference &R,
                          llvm::raw_ostream &OS) {
  llvm::SmallString<64> Path = R.getRelativeFilePath(CurrentPath);
  // Paths in Markdown use POSIX separators.
  llvm::sys::path::native(Path, llvm::sys::path::Style::posix);
  llvm::sys::path::append(Path, llvm::sys::path::Style::posix,
                          R.getFileBaseName() + ".md");
  OS << "[" << R.Name << "](" << Path << ")";
}

static void genMarkdown(const ClangDocContext &CDCtx, const EnumInfo &I,
                        llvm::raw_ostream &OS) {
  if (I.Scoped)
    writeLine("| enum class " + I.Name + " |", OS);
  else
    writeLine("| enum " + I.Name + " |", OS);
  writeLine("--", OS);

  std::string Buffer;
  llvm::raw_string_ostream Members(Buffer);
  if (!I.Members.empty())
    for (const auto &N : I.Members)
      Members << "| " << N.Name << " |\n";
  writeLine(Members.str(), OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  for (const auto &C : I.Description)
    writeDescription(C, OS);
}

static void genMarkdown(const ClangDocContext &CDCtx, const FunctionInfo &I,
                        llvm::raw_ostream &OS) {
  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);
  bool First = true;
  for (const auto &N : I.Params) {
    if (!First)
      Stream << ", ";
    Stream << N.Type.Name + " " + N.Name;
    First = false;
  }
  writeHeader(I.Name, 3, OS);
  std::string Access = getAccessSpelling(I.Access).str();
  if (Access != "")
    writeLine(genItalic(Access + " " + I.ReturnType.Type.Name + " " + I.Name +
                        "(" + Stream.str() + ")"),
              OS);
  else
    writeLine(genItalic(I.ReturnType.Type.Name + " " + I.Name + "(" +
                        Stream.str() + ")"),
              OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  for (const auto &C : I.Description)
    writeDescription(C, OS);
}

static void genMarkdown(const ClangDocContext &CDCtx, const NamespaceInfo &I,
                        llvm::raw_ostream &OS) {
  if (I.Name == "")
    writeHeader("Global Namespace", 1, OS);
  else
    writeHeader("namespace " + I.Name, 1, OS);
  writeNewLine(OS);

  if (!I.Description.empty()) {
    for (const auto &C : I.Description)
      writeDescription(C, OS);
    writeNewLine(OS);
  }

  llvm::SmallString<64> BasePath = I.getRelativeFilePath("");

  if (!I.Children.Namespaces.empty()) {
    writeHeader("Namespaces", 2, OS);
    for (const auto &R : I.Children.Namespaces) {
      OS << "* ";
      writeNameLink(BasePath, R, OS);
      OS << "\n";
    }
    writeNewLine(OS);
  }

  if (!I.Children.Records.empty()) {
    writeHeader("Records", 2, OS);
    for (const auto &R : I.Children.Records) {
      OS << "* ";
      writeNameLink(BasePath, R, OS);
      OS << "\n";
    }
    writeNewLine(OS);
  }

  if (!I.Children.Functions.empty()) {
    writeHeader("Functions", 2, OS);
    for (const auto &F : I.Children.Functions)
      genMarkdown(CDCtx, F, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Enums.empty()) {
    writeHeader("Enums", 2, OS);
    for (const auto &E : I.Children.Enums)
      genMarkdown(CDCtx, E, OS);
    writeNewLine(OS);
  }
}

static void genMarkdown(const ClangDocContext &CDCtx, const RecordInfo &I,
                        llvm::raw_ostream &OS) {
  writeHeader(getTagType(I.TagType) + " " + I.Name, 1, OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  if (!I.Description.empty()) {
    for (const auto &C : I.Description)
      writeDescription(C, OS);
    writeNewLine(OS);
  }

  std::string Parents = genReferenceList(I.Parents);
  std::string VParents = genReferenceList(I.VirtualParents);
  if (!Parents.empty() || !VParents.empty()) {
    if (Parents.empty())
      writeLine("Inherits from " + VParents, OS);
    else if (VParents.empty())
      writeLine("Inherits from " + Parents, OS);
    else
      writeLine("Inherits from " + Parents + ", " + VParents, OS);
    writeNewLine(OS);
  }

  if (!I.Members.empty()) {
    writeHeader("Members", 2, OS);
    for (const auto &Member : I.Members) {
      std::string Access = getAccessSpelling(Member.Access).str();
      if (Access != "")
        writeLine(Access + " " + Member.Type.Name + " " + Member.Name, OS);
      else
        writeLine(Member.Type.Name + " " + Member.Name, OS);
    }
    writeNewLine(OS);
  }

  if (!I.Children.Records.empty()) {
    writeHeader("Records", 2, OS);
    for (const auto &R : I.Children.Records)
      writeLine(R.Name, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Functions.empty()) {
    writeHeader("Functions", 2, OS);
    for (const auto &F : I.Children.Functions)
      genMarkdown(CDCtx, F, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Enums.empty()) {
    writeHeader("Enums", 2, OS);
    for (const auto &E : I.Children.Enums)
      genMarkdown(CDCtx, E, OS);
    writeNewLine(OS);
  }
}

static void genMarkdown(const ClangDocContext &CDCtx, const TypedefInfo &I,
                        llvm::raw_ostream &OS) {
  // TODO support typedefs in markdown.
}

static void serializeReference(llvm::raw_fd_ostream &OS, Index &I, int Level) {
  // Write out the heading level starting at ##
  OS << "##" << std::string(Level, '#') << " ";
  writeNameLink("", I, OS);
  OS << "\n";
}

static llvm::Error serializeIndex(ClangDocContext &CDCtx) {
  std::error_code FileErr;
  llvm::SmallString<128> FilePath;
  llvm::sys::path::native(CDCtx.OutDirectory, FilePath);
  llvm::sys::path::append(FilePath, "all_files.md");
  llvm::raw_fd_ostream OS(FilePath, FileErr, llvm::sys::fs::OF_None);
  if (FileErr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error creating index file: " +
                                       FileErr.message());

  CDCtx.Idx.sort();
  OS << "# All Files";
  if (!CDCtx.ProjectName.empty())
    OS << " for " << CDCtx.ProjectName;
  OS << "\n\n";

  for (auto C : CDCtx.Idx.Children)
    serializeReference(OS, C, 0);

  return llvm::Error::success();
}

static llvm::Error genIndex(ClangDocContext &CDCtx) {
  std::error_code FileErr;
  llvm::SmallString<128> FilePath;
  llvm::sys::path::native(CDCtx.OutDirectory, FilePath);
  llvm::sys::path::append(FilePath, "index.md");
  llvm::raw_fd_ostream OS(FilePath, FileErr, llvm::sys::fs::OF_None);
  if (FileErr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error creating index file: " +
                                       FileErr.message());
  CDCtx.Idx.sort();
  OS << "# " << CDCtx.ProjectName << " C/C++ Reference\n\n";
  for (auto C : CDCtx.Idx.Children) {
    if (!C.Children.empty()) {
      const char *Type;
      switch (C.RefType) {
      case InfoType::IT_namespace:
        Type = "Namespace";
        break;
      case InfoType::IT_record:
        Type = "Type";
        break;
      case InfoType::IT_enum:
        Type = "Enum";
        break;
      case InfoType::IT_function:
        Type = "Function";
        break;
      case InfoType::IT_typedef:
        Type = "Typedef";
        break;
      case InfoType::IT_default:
        Type = "Other";
      }
      OS << "* " << Type << ": [" << C.Name << "](";
      if (!C.Path.empty())
        OS << C.Path << "/";
      OS << C.Name << ")\n";
    }
  }
  return llvm::Error::success();
}

/// Generator for Markdown documentation.
class MDGenerator : public Generator {
public:
  static const char *Format;

  llvm::Error generateDocs(StringRef RootDir,
                           llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                           const ClangDocContext &CDCtx) override;
  llvm::Error createResources(ClangDocContext &CDCtx) override;
  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                 const ClangDocContext &CDCtx) override;
};

const char *MDGenerator::Format = "md";

llvm::Error
MDGenerator::generateDocs(StringRef RootDir,
                          llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                          const ClangDocContext &CDCtx) {
  // Track which directories we already tried to create.
  llvm::StringSet<> CreatedDirs;

  // Collect all output by file name and create the necessary directories.
  llvm::StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    doc::Info *Info = Group.getValue().get();

    llvm::SmallString<128> Path;
    llvm::sys::path::native(RootDir, Path);
    llvm::sys::path::append(Path, Info->getRelativeFilePath(""));
    if (CreatedDirs.find(Path) == CreatedDirs.end()) {
      if (std::error_code Err = llvm::sys::fs::create_directories(Path);
          Err != std::error_code()) {
        return llvm::createStringError(Err, "Failed to create directory '%s'.",
                                       Path.c_str());
      }
      CreatedDirs.insert(Path);
    }

    llvm::sys::path::append(Path, Info->getFileBaseName() + ".md");
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

llvm::Error MDGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                            const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    genMarkdown(CDCtx, *static_cast<clang::doc::NamespaceInfo *>(I), OS);
    break;
  case InfoType::IT_record:
    genMarkdown(CDCtx, *static_cast<clang::doc::RecordInfo *>(I), OS);
    break;
  case InfoType::IT_enum:
    genMarkdown(CDCtx, *static_cast<clang::doc::EnumInfo *>(I), OS);
    break;
  case InfoType::IT_function:
    genMarkdown(CDCtx, *static_cast<clang::doc::FunctionInfo *>(I), OS);
    break;
  case InfoType::IT_typedef:
    genMarkdown(CDCtx, *static_cast<clang::doc::TypedefInfo *>(I), OS);
    break;
  case InfoType::IT_default:
    return createStringError(llvm::inconvertibleErrorCode(),
                             "unexpected InfoType");
  }
  return llvm::Error::success();
}

llvm::Error MDGenerator::createResources(ClangDocContext &CDCtx) {
  // Write an all_files.md
  auto Err = serializeIndex(CDCtx);
  if (Err)
    return Err;

  // Generate the index page.
  Err = genIndex(CDCtx);
  if (Err)
    return Err;

  return llvm::Error::success();
}

static GeneratorRegistry::Add<MDGenerator> MD(MDGenerator::Format,
                                              "Generator for MD output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int MDGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
