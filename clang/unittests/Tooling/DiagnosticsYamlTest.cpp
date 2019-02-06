//===- unittests/Tooling/DiagnosticsYamlTest.cpp - Serialization tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for serialization of Diagnostics.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DiagnosticsYaml.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang::tooling;
using clang::tooling::Diagnostic;

static DiagnosticMessage makeMessage(const std::string &Message, int FileOffset,
                                     const std::string &FilePath) {
  DiagnosticMessage DiagMessage;
  DiagMessage.Message = Message;
  DiagMessage.FileOffset = FileOffset;
  DiagMessage.FilePath = FilePath;
  return DiagMessage;
}

static Diagnostic makeDiagnostic(StringRef DiagnosticName,
                                 const std::string &Message, int FileOffset,
                                 const std::string &FilePath,
                                 const StringMap<Replacements> &Fix) {
  return Diagnostic(DiagnosticName, makeMessage(Message, FileOffset, FilePath),
                    Fix, {}, Diagnostic::Warning, "path/to/build/directory");
}

TEST(DiagnosticsYamlTest, serializesDiagnostics) {
  TranslationUnitDiagnostics TUD;
  TUD.MainSourceFile = "path/to/source.cpp";

  StringMap<Replacements> Fix1 = {
      {"path/to/source.cpp",
       Replacements({"path/to/source.cpp", 100, 12, "replacement #1"})}};
  TUD.Diagnostics.push_back(makeDiagnostic("diagnostic#1", "message #1", 55,
                                           "path/to/source.cpp", Fix1));

  StringMap<Replacements> Fix2 = {
      {"path/to/header.h",
       Replacements({"path/to/header.h", 62, 2, "replacement #2"})}};
  TUD.Diagnostics.push_back(makeDiagnostic("diagnostic#2", "message #2", 60,
                                           "path/to/header.h", Fix2));

  TUD.Diagnostics.push_back(makeDiagnostic("diagnostic#3", "message #3", 72,
                                           "path/to/source2.cpp", {}));
  TUD.Diagnostics.back().Notes.push_back(
      makeMessage("Note1", 88, "path/to/note1.cpp"));
  TUD.Diagnostics.back().Notes.push_back(
      makeMessage("Note2", 99, "path/to/note2.cpp"));

  std::string YamlContent;
  raw_string_ostream YamlContentStream(YamlContent);

  yaml::Output YAML(YamlContentStream);
  YAML << TUD;

  EXPECT_EQ("---\n"
            "MainSourceFile:  'path/to/source.cpp'\n"
            "Diagnostics:     \n"
            "  - DiagnosticName:  'diagnostic#1\'\n"
            "    Message:         'message #1'\n"
            "    FileOffset:      55\n"
            "    FilePath:        'path/to/source.cpp'\n"
            "    Replacements:    \n"
            "      - FilePath:        'path/to/source.cpp'\n"
            "        Offset:          100\n"
            "        Length:          12\n"
            "        ReplacementText: 'replacement #1'\n"
            "  - DiagnosticName:  'diagnostic#2'\n"
            "    Message:         'message #2'\n"
            "    FileOffset:      60\n"
            "    FilePath:        'path/to/header.h'\n"
            "    Replacements:    \n"
            "      - FilePath:        'path/to/header.h'\n"
            "        Offset:          62\n"
            "        Length:          2\n"
            "        ReplacementText: 'replacement #2'\n"
            "  - DiagnosticName:  'diagnostic#3'\n"
            "    Message:         'message #3'\n"
            "    FileOffset:      72\n"
            "    FilePath:        'path/to/source2.cpp'\n"
            "    Notes:           \n"
            "      - Message:         Note1\n"
            "        FilePath:        'path/to/note1.cpp'\n"
            "        FileOffset:      88\n"
            "      - Message:         Note2\n"
            "        FilePath:        'path/to/note2.cpp'\n"
            "        FileOffset:      99\n"
            "    Replacements:    []\n"
            "...\n",
            YamlContentStream.str());
}

TEST(DiagnosticsYamlTest, deserializesDiagnostics) {
  std::string YamlContent = "---\n"
                            "MainSourceFile:  path/to/source.cpp\n"
                            "Diagnostics:     \n"
                            "  - DiagnosticName:  'diagnostic#1'\n"
                            "    Message:         'message #1'\n"
                            "    FileOffset:      55\n"
                            "    FilePath:        path/to/source.cpp\n"
                            "    Replacements:    \n"
                            "      - FilePath:        path/to/source.cpp\n"
                            "        Offset:          100\n"
                            "        Length:          12\n"
                            "        ReplacementText: 'replacement #1'\n"
                            "  - DiagnosticName:  'diagnostic#2'\n"
                            "    Message:         'message #2'\n"
                            "    FileOffset:      60\n"
                            "    FilePath:        path/to/header.h\n"
                            "    Replacements:    \n"
                            "      - FilePath:        path/to/header.h\n"
                            "        Offset:          62\n"
                            "        Length:          2\n"
                            "        ReplacementText: 'replacement #2'\n"
                            "  - DiagnosticName:  'diagnostic#3'\n"
                            "    Message:         'message #3'\n"
                            "    FileOffset:      98\n"
                            "    FilePath:        path/to/source.cpp\n"
                            "    Notes:\n"
                            "      - Message:         Note1\n"
                            "        FilePath:        'path/to/note1.cpp'\n"
                            "        FileOffset:      66\n"
                            "      - Message:         Note2\n"
                            "        FilePath:        'path/to/note2.cpp'\n"
                            "        FileOffset:      77\n"
                            "    Replacements:    []\n"
                            "...\n";
  TranslationUnitDiagnostics TUDActual;
  yaml::Input YAML(YamlContent);
  YAML >> TUDActual;

  ASSERT_FALSE(YAML.error());
  ASSERT_EQ(3u, TUDActual.Diagnostics.size());
  EXPECT_EQ("path/to/source.cpp", TUDActual.MainSourceFile);

  auto getFixes = [](const StringMap<Replacements> &Fix) {
    std::vector<Replacement> Fixes;
    for (auto &Replacements : Fix) {
      for (auto &Replacement : Replacements.second) {
        Fixes.push_back(Replacement);
      }
    }
    return Fixes;
  };

  Diagnostic D1 = TUDActual.Diagnostics[0];
  EXPECT_EQ("diagnostic#1", D1.DiagnosticName);
  EXPECT_EQ("message #1", D1.Message.Message);
  EXPECT_EQ(55u, D1.Message.FileOffset);
  EXPECT_EQ("path/to/source.cpp", D1.Message.FilePath);
  std::vector<Replacement> Fixes1 = getFixes(D1.Fix);
  ASSERT_EQ(1u, Fixes1.size());
  EXPECT_EQ("path/to/source.cpp", Fixes1[0].getFilePath());
  EXPECT_EQ(100u, Fixes1[0].getOffset());
  EXPECT_EQ(12u, Fixes1[0].getLength());
  EXPECT_EQ("replacement #1", Fixes1[0].getReplacementText());

  Diagnostic D2 = TUDActual.Diagnostics[1];
  EXPECT_EQ("diagnostic#2", D2.DiagnosticName);
  EXPECT_EQ("message #2", D2.Message.Message);
  EXPECT_EQ(60u, D2.Message.FileOffset);
  EXPECT_EQ("path/to/header.h", D2.Message.FilePath);
  std::vector<Replacement> Fixes2 = getFixes(D2.Fix);
  ASSERT_EQ(1u, Fixes2.size());
  EXPECT_EQ("path/to/header.h", Fixes2[0].getFilePath());
  EXPECT_EQ(62u, Fixes2[0].getOffset());
  EXPECT_EQ(2u, Fixes2[0].getLength());
  EXPECT_EQ("replacement #2", Fixes2[0].getReplacementText());

  Diagnostic D3 = TUDActual.Diagnostics[2];
  EXPECT_EQ("diagnostic#3", D3.DiagnosticName);
  EXPECT_EQ("message #3", D3.Message.Message);
  EXPECT_EQ(98u, D3.Message.FileOffset);
  EXPECT_EQ("path/to/source.cpp", D3.Message.FilePath);
  EXPECT_EQ(2u, D3.Notes.size());
  EXPECT_EQ("Note1", D3.Notes[0].Message);
  EXPECT_EQ(66u, D3.Notes[0].FileOffset);
  EXPECT_EQ("path/to/note1.cpp", D3.Notes[0].FilePath);
  EXPECT_EQ("Note2", D3.Notes[1].Message);
  EXPECT_EQ(77u, D3.Notes[1].FileOffset);
  EXPECT_EQ("path/to/note2.cpp", D3.Notes[1].FilePath);
  std::vector<Replacement> Fixes3 = getFixes(D3.Fix);
  EXPECT_TRUE(Fixes3.empty());
}
