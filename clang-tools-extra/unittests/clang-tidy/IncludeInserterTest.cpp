//===---- IncludeInserterTest.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/IncludeInserter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "ClangTidyTest.h"
#include "gtest/gtest.h"

// FIXME: Canonicalize paths correctly on windows.
// Currently, adding virtual files will canonicalize the paths before
// storing the virtual entries.
// When resolving virtual entries in the FileManager, the paths (for
// example coming from a #include directive) are not canonicalized
// to native paths; thus, the virtual file is not found.
// This needs to be fixed in the FileManager before we can make
// clang-tidy tests work.
#if !defined(_WIN32)

namespace clang {
namespace tidy {
namespace {

class IncludeInserterCheckBase : public ClangTidyCheck {
public:
  IncludeInserterCheckBase(StringRef CheckName, ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context) {}

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override {
    Inserter = llvm::make_unique<utils::IncludeInserter>(
        SM, getLangOpts(), utils::IncludeSorter::IS_Google);
    PP->addPPCallbacks(Inserter->CreatePPCallbacks());
  }

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(ast_matchers::declStmt().bind("stmt"), this);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    auto Diag = diag(Result.Nodes.getNodeAs<DeclStmt>("stmt")->getBeginLoc(),
                     "foo, bar");
    for (StringRef header : HeadersToInclude()) {
      auto Fixit = Inserter->CreateIncludeInsertion(
          Result.SourceManager->getMainFileID(), header, IsAngledInclude());
      if (Fixit) {
        Diag << *Fixit;
      }
    }
  }

  virtual std::vector<StringRef> HeadersToInclude() const = 0;
  virtual bool IsAngledInclude() const = 0;

  std::unique_ptr<utils::IncludeInserter> Inserter;
};

class NonSystemHeaderInserterCheck : public IncludeInserterCheckBase {
public:
  NonSystemHeaderInserterCheck(StringRef CheckName, ClangTidyContext *Context)
      : IncludeInserterCheckBase(CheckName, Context) {}

  std::vector<StringRef> HeadersToInclude() const override {
    return {"path/to/header.h"};
  }
  bool IsAngledInclude() const override { return false; }
};

class EarlyInAlphabetHeaderInserterCheck : public IncludeInserterCheckBase {
public:
  EarlyInAlphabetHeaderInserterCheck(StringRef CheckName, ClangTidyContext *Context)
      : IncludeInserterCheckBase(CheckName, Context) {}

  std::vector<StringRef> HeadersToInclude() const override {
    return {"a/header.h"};
  }
  bool IsAngledInclude() const override { return false; }
};

class MultipleHeaderInserterCheck : public IncludeInserterCheckBase {
public:
  MultipleHeaderInserterCheck(StringRef CheckName, ClangTidyContext *Context)
      : IncludeInserterCheckBase(CheckName, Context) {}

  std::vector<StringRef> HeadersToInclude() const override {
    return {"path/to/header.h", "path/to/header2.h", "path/to/header.h"};
  }
  bool IsAngledInclude() const override { return false; }
};

class CSystemIncludeInserterCheck : public IncludeInserterCheckBase {
public:
  CSystemIncludeInserterCheck(StringRef CheckName, ClangTidyContext *Context)
      : IncludeInserterCheckBase(CheckName, Context) {}

  std::vector<StringRef> HeadersToInclude() const override {
    return {"stdlib.h"};
  }
  bool IsAngledInclude() const override { return true; }
};

class CXXSystemIncludeInserterCheck : public IncludeInserterCheckBase {
public:
  CXXSystemIncludeInserterCheck(StringRef CheckName, ClangTidyContext *Context)
      : IncludeInserterCheckBase(CheckName, Context) {}

  std::vector<StringRef> HeadersToInclude() const override { return {"set"}; }
  bool IsAngledInclude() const override { return true; }
};

template <typename Check>
std::string runCheckOnCode(StringRef Code, StringRef Filename) {
  std::vector<ClangTidyError> Errors;
  return test::runCheckOnCode<Check>(Code, &Errors, Filename, None,
                                     ClangTidyOptions(),
                                     {// Main file include
                                      {"clang_tidy/tests/"
                                       "insert_includes_test_header.h",
                                       "\n"},
                                      // Non system headers
                                      {"a/header.h", "\n"},
                                      {"path/to/a/header.h", "\n"},
                                      {"path/to/z/header.h", "\n"},
                                      {"path/to/header.h", "\n"},
                                      {"path/to/header2.h", "\n"},
                                      // Fake system headers.
                                      {"stdlib.h", "\n"},
                                      {"unistd.h", "\n"},
                                      {"list", "\n"},
                                      {"map", "\n"},
                                      {"set", "\n"},
                                      {"vector", "\n"}});
}

TEST(IncludeInserterTest, InsertAfterLastNonSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"
#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_input2.cc"));
}

TEST(IncludeInserterTest, InsertMultipleIncludesAndDeduplicate) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"
#include "path/to/header.h"
#include "path/to/header2.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<MultipleHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_input2.cc"));
}

TEST(IncludeInserterTest, InsertBeforeFirstNonSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/z/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/header.h"
#include "path/to/z/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_input2.cc"));
}

TEST(IncludeInserterTest, InsertBetweenNonSystemIncludes) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"
#include "path/to/z/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"
#include "path/to/header.h"
#include "path/to/z/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_input2.cc"));
}

TEST(IncludeInserterTest, NonSystemIncludeAlreadyIncluded) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"
#include "path/to/header.h"
#include "path/to/z/header.h"

void foo() {
  int a = 0;
})";
  EXPECT_EQ(PreCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                         PreCode, "clang_tidy/tests/"
                                  "insert_includes_test_input2.cc"));
}

TEST(IncludeInserterTest, InsertNonSystemIncludeAfterLastCXXSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertNonSystemIncludeAfterMainFileInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<NonSystemHeaderInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeAfterLastCXXSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <list>
#include <map>
#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeBeforeFirstCXXSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <vector>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <set>
#include <vector>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeBetweenCXXSystemIncludes) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <vector>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <set>
#include <vector>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeAfterMainFileInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeAfterCSystemInclude) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <stdlib.h>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <stdlib.h>

#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCXXSystemIncludeBeforeNonSystemInclude) {
  const char *PreCode = R"(
#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "devtools/cymbal/clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertCSystemIncludeBeforeCXXSystemInclude) {
  const char *PreCode = R"(
#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";
  const char *PostCode = R"(
#include <stdlib.h>

#include <set>

#include "path/to/a/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CSystemIncludeInserterCheck>(
                          PreCode, "devtools/cymbal/clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, InsertIncludeIfThereWasNoneBefore) {
  const char *PreCode = R"(
void foo() {
  int a = 0;
})";
  const char *PostCode = R"(#include <set>


void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<CXXSystemIncludeInserterCheck>(
                          PreCode, "devtools/cymbal/clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, DontInsertDuplicateIncludeEvenIfMiscategorized) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <set>
#include <vector>

#include "a/header.h"
#include "path/to/a/header.h"
#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <set>
#include <vector>

#include "a/header.h"
#include "path/to/a/header.h"
#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<EarlyInAlphabetHeaderInserterCheck>(
                          PreCode, "workspace_folder/clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

TEST(IncludeInserterTest, HandleOrderInSubdirectory) {
  const char *PreCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <set>
#include <vector>

#include "path/to/a/header.h"
#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  const char *PostCode = R"(
#include "clang_tidy/tests/insert_includes_test_header.h"

#include <map>
#include <set>
#include <vector>

#include "a/header.h"
#include "path/to/a/header.h"
#include "path/to/header.h"

void foo() {
  int a = 0;
})";

  EXPECT_EQ(PostCode, runCheckOnCode<EarlyInAlphabetHeaderInserterCheck>(
                          PreCode, "workspace_folder/clang_tidy/tests/"
                                   "insert_includes_test_header.cc"));
}

} // anonymous namespace
} // namespace tidy
} // namespace clang

#endif
