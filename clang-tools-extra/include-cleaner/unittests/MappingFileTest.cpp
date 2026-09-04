//===--- MappingFileTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/MappingFile.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace clang::include_cleaner {
namespace {

using ::llvm::Failed;
using ::llvm::HasValue;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::ResultOf;
using ::testing::UnorderedElementsAre;

// Write content to a temporary file, returning the path.
std::string writeTempFile(llvm::StringRef Content) {
  llvm::SmallString<256> Path;
  std::error_code EC =
      llvm::sys::fs::createTemporaryFile("mapping", "imp", Path);
  EXPECT_FALSE(EC);
  std::error_code WriteEC;
  llvm::raw_fd_ostream OS(Path, WriteEC);
  EXPECT_FALSE(WriteEC);
  OS << Content;
  OS.close();
  return Path.str().str();
}

// StringMap entries lack the first_type/second_type typedefs that Pair
// requires, so convert to a vector of std::pair for matching.
std::vector<std::pair<std::string, std::string>>
toPairs(const llvm::StringMap<std::string> &M) {
  std::vector<std::pair<std::string, std::string>> Result;
  for (const auto &E : M)
    Result.emplace_back(E.getKey().str(), E.getValue());
  return Result;
}

// Convert backslashes to forward slashes for safe embedding in YAML strings.
// On Windows, raw paths would produce unrecognized escape sequences (e.g. \U).
std::string toYAMLPath(llvm::StringRef Path) {
  return llvm::sys::path::convert_to_slash(Path);
}

TEST(MappingFileTest, EmptyArray) {
  std::string Path = writeTempFile("[]");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings, IsEmpty()),
          Field("SymbolMappings", &MappingFile::SymbolMappings, IsEmpty()))));
}

TEST(MappingFileTest, IncludeMapping_AngleBrackets) {
  std::string Path = writeTempFile(
      R"([{"include": ["<private.h>", "private", "<public.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings,
                ResultOf(toPairs, UnorderedElementsAre(
                                      Pair("private.h", "<public.h>")))),
          Field("SymbolMappings", &MappingFile::SymbolMappings, IsEmpty()))));
}

TEST(MappingFileTest, IncludeMapping_QuotedHeaders) {
  std::string Path = writeTempFile(
      R"([{"include": ["\"private.h\"", "private", "\"public.h\"", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("IncludeMappings", &MappingFile::IncludeMappings,
                     ResultOf(toPairs, UnorderedElementsAre(Pair(
                                           "private.h", "\"public.h\""))))));
}

TEST(MappingFileTest, SymbolMapping) {
  std::string Path = writeTempFile(
      R"([{"symbol": ["NULL", "private", "<stddef.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("SymbolMappings", &MappingFile::SymbolMappings,
                ResultOf(toPairs,
                         UnorderedElementsAre(Pair("NULL", "<stddef.h>")))),
          Field("IncludeMappings", &MappingFile::IncludeMappings, IsEmpty()))));
}

TEST(MappingFileTest, SymbolMapping_QualifiedName) {
  std::string Path = writeTempFile(
      R"([{"symbol": ["std::string", "private", "<string>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("SymbolMappings", &MappingFile::SymbolMappings,
                     ResultOf(toPairs, UnorderedElementsAre(
                                           Pair("std::string", "<string>"))))));
}

TEST(MappingFileTest, MultipleEntries) {
  std::string Path = writeTempFile(R"([
    {"include": ["<a.h>", "private", "<b.h>", "public"]},
    {"symbol": ["MyType", "private", "<mytype.h>", "public"]},
    {"include": ["<c.h>", "private", "<d.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>"),
                                                       Pair("c.h", "<d.h>")))),
          Field("SymbolMappings", &MappingFile::SymbolMappings,
                ResultOf(toPairs, UnorderedElementsAre(
                                      Pair("MyType", "<mytype.h>")))))));
}

TEST(MappingFileTest, UnquotedPrivatePublic) {
  // Traditional IWYU format with unquoted private/public visibility values.
  std::string Path = writeTempFile(
      "[{\"include\": [\"<private.h>\", private, \"<public.h>\", public]}]");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("IncludeMappings", &MappingFile::IncludeMappings,
                     ResultOf(toPairs, UnorderedElementsAre(
                                           Pair("private.h", "<public.h>"))))));
}

TEST(MappingFileTest, HashLineComment) {
  std::string Path = writeTempFile(R"(
  # This is a comment
  [
    # Another comment
    {"include": ["<a.h>", "private", "<b.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>"))))));
}

TEST(MappingFileTest, HashCommentAfterValue) {
  std::string Path = writeTempFile(
      "[{\"include\": [\"<a.h>\", \"private\", \"<b.h>\", \"public\"]}] # end");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>"))))));
}

TEST(MappingFileTest, RefEntry) {
  std::string RefPath =
      writeTempFile(R"([{"symbol": ["Foo", "private", "<foo.h>", "public"]}])");
  std::string MainPath =
      writeTempFile(R"([{"ref": ")" + toYAMLPath(RefPath) + R"("}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({MainPath}),
      HasValue(Field(
          "SymbolMappings", &MappingFile::SymbolMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("Foo", "<foo.h>"))))));
}

TEST(MappingFileTest, MultiplePaths) {
  std::string Path1 = writeTempFile(
      R"([{"include": ["<a.h>", "private", "<b.h>", "public"]}])");
  std::string Path2 =
      writeTempFile(R"([{"symbol": ["X", "private", "<x.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path1, Path2}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>")))),
          Field("SymbolMappings", &MappingFile::SymbolMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("X", "<x.h>")))))));
}

TEST(MappingFileTest, RegexPattern_AngleBrackets) {
  // "@<AE/.*>" is a regex pattern stored in IncludeRegexPatterns.
  std::string Path = writeTempFile(
      R"([{"include": ["@<AE/.*>", "private", "<Carbon/Carbon.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings, IsEmpty()),
          Field("IncludeRegexPatterns", &MappingFile::IncludeRegexPatterns,
                ElementsAre(Pair("AE/.*", "<Carbon/Carbon.h>"))))));
}

TEST(MappingFileTest, RegexPattern_QuotedHeader) {
  // "@\"foo/.*\"" extracts regex from quoted form.
  std::string Path = writeTempFile("[{\"include\": [\"@\\\"foo/.*\\\"\", "
                                   "\"private\", \"<foo.h>\", \"public\"]}]");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("IncludeRegexPatterns", &MappingFile::IncludeRegexPatterns,
                     ElementsAre(Pair("foo/.*", "<foo.h>")))));
}

TEST(MappingFileTest, RegexPattern_FrameworkStyle) {
  // Pattern "AE/.*" should be stored as-is; framework path matching is
  // handled at runtime in PragmaIncludes, not during parsing.
  std::string Path = writeTempFile(
      R"([{"include": ["@<AE/.*>", "private", "<Carbon/Carbon.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("IncludeRegexPatterns", &MappingFile::IncludeRegexPatterns,
                     ElementsAre(Pair("AE/.*", "<Carbon/Carbon.h>")))));
}

TEST(MappingFileTest, RegexPattern_MultipleFrameworks) {
  // Multiple regex patterns for different frameworks.
  std::string Path = writeTempFile(R"([
    {"include": ["@<AE/.*>",         "private", "<Carbon/Carbon.h>", "public"]},
    {"include": ["@<CarbonCore/.*>", "private", "<Carbon/Carbon.h>", "public"]},
    {"include": ["@<HIToolbox/.*>",  "private", "<Carbon/Carbon.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field("IncludeRegexPatterns", &MappingFile::IncludeRegexPatterns,
                     ElementsAre(Pair("AE/.*", "<Carbon/Carbon.h>"),
                                 Pair("CarbonCore/.*", "<Carbon/Carbon.h>"),
                                 Pair("HIToolbox/.*", "<Carbon/Carbon.h>")))));
}

TEST(MappingFileTest, RegexAndExactInSameFile) {
  std::string Path = writeTempFile(R"([
    {"include": ["<exact.h>", "private", "<public.h>", "public"]},
    {"include": ["@<regex/.*>", "private", "<public.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings,
                ResultOf(toPairs,
                         UnorderedElementsAre(Pair("exact.h", "<public.h>")))),
          Field("IncludeRegexPatterns", &MappingFile::IncludeRegexPatterns,
                ElementsAre(Pair("regex/.*", "<public.h>"))))));
}

TEST(MappingFileTest, TrailingComma) {
  // YAML (and IWYU) allow a trailing comma after the last element.
  std::string Path = writeTempFile(R"([
    {"include": ["<a.h>", "private", "<b.h>", "public"]},
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>"))))));
}

TEST(MappingFileTest, WrongFieldCount_TooFew) {
  // Entry with 3 scalars is silently skipped; other valid entries are kept.
  std::string Path = writeTempFile(R"([
    {"include": ["<a.h>", "private", "<b.h>"]},
    {"include": ["<c.h>", "private", "<d.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("c.h", "<d.h>"))))));
}

TEST(MappingFileTest, WrongFieldCount_TooMany) {
  // Entry with 5 scalars is silently skipped; other valid entries are kept.
  std::string Path = writeTempFile(R"([
    {"include": ["<a.h>", "private", "<b.h>", "public", "extra"]},
    {"include": ["<c.h>", "private", "<d.h>", "public"]}
  ])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("c.h", "<d.h>"))))));
}

TEST(MappingFileTest, RefToNonexistentFile) {
  std::string Path =
      writeTempFile(R"([{"ref": "/nonexistent/path/to/ref.imp"}])");
  EXPECT_THAT_EXPECTED(parseMappingFiles({Path}), Failed());
}

TEST(MappingFileTest, DuplicateKeys_FirstFilePriority) {
  // When two files map the same key, the first-listed file takes priority.
  std::string Path1 = writeTempFile(
      R"([{"include": ["<a.h>", "private", "<first.h>", "public"]}])");
  std::string Path2 = writeTempFile(
      R"([{"include": ["<a.h>", "private", "<second.h>", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path1, Path2}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<first.h>"))))));
}

TEST(MappingFileTest, MultipleRefs) {
  // A file may contain multiple "ref" entries; all referenced files are merged.
  std::string RefPath1 = writeTempFile(
      R"([{"include": ["<a.h>", "private", "<b.h>", "public"]}])");
  std::string RefPath2 =
      writeTempFile(R"([{"symbol": ["X", "private", "<x.h>", "public"]}])");
  std::string MainPath =
      writeTempFile(R"([{"ref": ")" + toYAMLPath(RefPath1) +
                    R"("}, {"ref": ")" + toYAMLPath(RefPath2) + R"("}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({MainPath}),
      HasValue(AllOf(
          Field("IncludeMappings", &MappingFile::IncludeMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>")))),
          Field("SymbolMappings", &MappingFile::SymbolMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("X", "<x.h>")))))));
}

TEST(MappingFileTest, RelativeRef) {
  // Relative ref paths are resolved against the referring file's directory.
  std::string RefPath =
      writeTempFile(R"([{"symbol": ["Bar", "private", "<bar.h>", "public"]}])");
  // Both temp files land in the same directory, so the bare filename is a
  // valid relative ref from the main file.
  std::string RefFilename = llvm::sys::path::filename(RefPath).str();
  std::string MainPath =
      writeTempFile(R"([{"ref": ")" + RefFilename + R"("}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({MainPath}),
      HasValue(Field(
          "SymbolMappings", &MappingFile::SymbolMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("Bar", "<bar.h>"))))));
}

TEST(MappingFileTest, CircularRef) {
  // Circular refs terminate via the Visited set; entries from all files are
  // merged.
  llvm::SmallString<256> PathA, PathB;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("mapping", "imp", PathA));
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("mapping", "imp", PathB));
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(PathA, EC);
    ASSERT_FALSE(EC);
    OS << "[{\"ref\": \"" << toYAMLPath(PathB)
       << "\"}, {\"symbol\": [\"A\", \"private\", \"<a.h>\", \"public\"]}]";
  }
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(PathB, EC);
    ASSERT_FALSE(EC);
    OS << "[{\"ref\": \"" << toYAMLPath(PathA)
       << "\"}, {\"symbol\": [\"B\", \"private\", \"<b.h>\", \"public\"]}]";
  }
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({PathA.str().str()}),
      HasValue(
          Field("SymbolMappings", &MappingFile::SymbolMappings,
                ResultOf(toPairs, UnorderedElementsAre(Pair("A", "<a.h>"),
                                                       Pair("B", "<b.h>"))))));
}

TEST(MappingFileTest, NonexistentFile) {
  EXPECT_THAT_EXPECTED(parseMappingFiles({"/nonexistent/path/to/file.imp"}),
                       Failed());
}

TEST(MappingFileTest, InvalidYAML) {
  std::string Path = writeTempFile("not yaml: [}");
  EXPECT_THAT_EXPECTED(parseMappingFiles({Path}), Failed());
}

TEST(MappingFileTest, ToWithoutDelimiters) {
  // If the "to" header has no delimiters, angle brackets are added.
  std::string Path =
      writeTempFile(R"([{"include": ["<a.h>", "private", "b.h", "public"]}])");
  ASSERT_THAT_EXPECTED(
      parseMappingFiles({Path}),
      HasValue(Field(
          "IncludeMappings", &MappingFile::IncludeMappings,
          ResultOf(toPairs, UnorderedElementsAre(Pair("a.h", "<b.h>"))))));
}

} // namespace
} // namespace clang::include_cleaner
