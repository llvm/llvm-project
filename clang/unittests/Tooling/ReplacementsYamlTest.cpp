//===- unittests/Tooling/ReplacementsYamlTest.cpp - Serialization tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for serialization of Replacements.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ReplacementsYaml.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang::tooling;

TEST(ReplacementsYamlTest, serializesReplacements) {

  TranslationUnitReplacements Doc;

  Doc.MainSourceFile = "/path/to/source.cpp";
  Doc.Replacements.emplace_back("/path/to/file1.h", 232, 56, "replacement #1");
  Doc.Replacements.emplace_back("/path/to/file2.h", 301, 2, "replacement #2");

  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);

  yaml::Output YAML(YamlContentStream);
  YAML << Doc;

  // NOTE: If this test starts to fail for no obvious reason, check whitespace.
  ASSERT_STREQ("---\n"
               "MainSourceFile:  '/path/to/source.cpp'\n"
               "Replacements:    \n" // Extra whitespace here!
               "  - FilePath:        '/path/to/file1.h'\n"
               "    Offset:          232\n"
               "    Length:          56\n"
               "    ReplacementText: 'replacement #1'\n"
               "  - FilePath:        '/path/to/file2.h'\n"
               "    Offset:          301\n"
               "    Length:          2\n"
               "    ReplacementText: 'replacement #2'\n"
               "...\n",
               YamlContentStream.str().c_str());
}

TEST(ReplacementsYamlTest, deserializesReplacements) {
  std::string YamlContent = "---\n"
                            "MainSourceFile:      /path/to/source.cpp\n"
                            "Replacements:\n"
                            "  - FilePath:        /path/to/file1.h\n"
                            "    Offset:          232\n"
                            "    Length:          56\n"
                            "    ReplacementText: 'replacement #1'\n"
                            "  - FilePath:        /path/to/file2.h\n"
                            "    Offset:          301\n"
                            "    Length:          2\n"
                            "    ReplacementText: 'replacement #2'\n"
                            "...\n";
  TranslationUnitReplacements DocActual;
  yaml::Input YAML(YamlContent);
  YAML >> DocActual;
  ASSERT_FALSE(YAML.error());
  ASSERT_EQ(2u, DocActual.Replacements.size());
  ASSERT_EQ("/path/to/source.cpp", DocActual.MainSourceFile);
  ASSERT_EQ("/path/to/file1.h", DocActual.Replacements[0].getFilePath());
  ASSERT_EQ(232u, DocActual.Replacements[0].getOffset());
  ASSERT_EQ(56u, DocActual.Replacements[0].getLength());
  ASSERT_EQ("replacement #1", DocActual.Replacements[0].getReplacementText());
  ASSERT_EQ("/path/to/file2.h", DocActual.Replacements[1].getFilePath());
  ASSERT_EQ(301u, DocActual.Replacements[1].getOffset());
  ASSERT_EQ(2u, DocActual.Replacements[1].getLength());
  ASSERT_EQ("replacement #2", DocActual.Replacements[1].getReplacementText());
}

TEST(ReplacementsYamlTest, deserializesWithoutContext) {
  // Make sure a doc can be read without the context field.
  std::string YamlContent = "---\n"
                            "MainSourceFile:      /path/to/source.cpp\n"
                            "Replacements:\n"
                            "  - FilePath:        target_file.h\n"
                            "    Offset:          1\n"
                            "    Length:          10\n"
                            "    ReplacementText: replacement\n"
                            "...\n";
  TranslationUnitReplacements DocActual;
  yaml::Input YAML(YamlContent);
  YAML >> DocActual;
  ASSERT_FALSE(YAML.error());
  ASSERT_EQ("/path/to/source.cpp", DocActual.MainSourceFile);
  ASSERT_EQ(1u, DocActual.Replacements.size());
  ASSERT_EQ("target_file.h", DocActual.Replacements[0].getFilePath());
  ASSERT_EQ(1u, DocActual.Replacements[0].getOffset());
  ASSERT_EQ(10u, DocActual.Replacements[0].getLength());
  ASSERT_EQ("replacement", DocActual.Replacements[0].getReplacementText());
}
