//===- unittests/Serialization/ThinBMIDeclsHashTest.cpp - CI tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class ThinBMIDeclsHashTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("modules-test", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  using PathType = llvm::SmallString<256>;

  PathType TestDir;

public:
  FileManager FMgr = FileManager{FileSystemOptions()};

  void addFile(StringRef Path, StringRef Contents) {
    ASSERT_TRUE(sys::path::is_absolute(Path));
    ASSERT_TRUE(Path.startswith(TestDir));

    std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  PathType getUniquePathInTestDir(StringRef Suffix) {
    PathType Pattern("%%-%%-%%-%%-%%-%%");
    Pattern.append(Suffix);
    llvm::sys::fs::createUniquePath(Pattern, Pattern, /*MakeAbsolute=*/false);

    PathType Result(TestDir);
    llvm::sys::path::append(Result, Pattern);
    return Result;
  }

  // Map from module name to BMI path.
  using ModuleMapTy = llvm::StringMap<std::string>;

  std::string GenerateModuleInterface(StringRef Contents,
                                      StringRef AdditionalArgs = StringRef()) {
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(new DiagnosticOptions());
    CreateInvocationOptions CIOpts;
    CIOpts.Diags = Diags;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();

    PathType InterfaceSourceName = getUniquePathInTestDir(".cppm");
    addFile(InterfaceSourceName, Contents);

    const char *Args[] = {"clang++",
                          "-std=c++20",
                          "--precompile",
                          AdditionalArgs.data(),
                          "-working-directory",
                          TestDir.c_str(),
                          "-I",
                          TestDir.c_str(),
                          InterfaceSourceName.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);

    CompilerInstance Instance;
    Instance.setDiagnostics(Diags.get());
    Instance.setInvocation(Invocation);

    PathType CacheBMIPath = getUniquePathInTestDir(".pcm");
    Instance.getFrontendOpts().OutputFile = (std::string)CacheBMIPath;
    GenerateThinModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());

    return (std::string)CacheBMIPath;
  }

  std::optional<uint64_t> getBMIHash(StringRef BMIPath) {
    return ASTReader::getBMIHash(BMIPath, FMgr);
  }

  bool CompareBMIHash(StringRef Contents1, StringRef Contents2) {
    auto BMIPath1 = GenerateModuleInterface(Contents1);
    auto BMIPath2 = GenerateModuleInterface(Contents2);

    std::optional<uint64_t> Hash1 = getBMIHash(BMIPath1);
    EXPECT_TRUE(Hash1);
    std::optional<uint64_t> Hash2 = getBMIHash(BMIPath2);
    EXPECT_TRUE(Hash2);

    return *Hash1 == *Hash2;
  }
};

// Test that:
// - the BMI hash won't change if we only touched the body of a non-inline
// function.
// - the BMI hash will change if we changed the interface of exported non-inline
// function.
// - the BMI hash will change if we add or delete exported functions.
// - the BMI hash won't change if we changed, add and delete non-exported and
// non-inline
//   function.
TEST_F(ThinBMIDeclsHashTest, BasicNonInlineFuncTest) {
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                             R"cpp(
export module a;
export int a() {
  return 44;
}
  )cpp"));

  // The interface should change if we change the function name.
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export int b() {
  return 43;
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export char a() {
  return 44;
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export int a(int v = 43) {
  return 44 + v;
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a(int v = 44) {
  return v;
}
  )cpp",
                              R"cpp(
export module a;
export int a(int v = 43) {
  return v;
}
  )cpp"));

  // Test that the comment won't change the interface.
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                             R"cpp(
export module a;
// New comments here.
export int a() {
  return 43;
}
  )cpp"));

  // Test that adding new exported functions may change the interface.
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export int a() {
  return 43;
}

export int a2() {
  return a() + 43;
}
  )cpp"));

  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                             R"cpp(
export module a;

int non_exported() { return 49; }

export int a() {
  return non_exported();
}
  )cpp"));

  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                             R"cpp(
export module a;

int non_exported() { return 99; }

export int a() {
  return non_exported();
}
  )cpp"));
}

// Tests that:
// - The interface shouldn't change if we changed the definition of the
// non-inline variables.
// - The interface shouldn change if we change the interface (type and name) of
// the exported
//   non-inline variables.
// - The interface shouldn't change if we change, add or remove non-exported and
// non-inline
//   variables.
TEST_F(ThinBMIDeclsHashTest, BasicNonInlineVarTest) {
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                             R"cpp(
export module a;
export int a = 45;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                              R"cpp(
export module a;
export short a = 43;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                              R"cpp(
export module a;
export double a = 43.0;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                              R"cpp(
export module a;
export int aa = 43;
  )cpp"));

  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                             R"cpp(
export module a;
int a_def();
export int a = a_def();
  )cpp"));

  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                             R"cpp(
export module a;
int a_def();
int a_non_exported = a_def();
export int a = a_non_exported;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                              R"cpp(
export module a;
export int a = 43;
export int another_exported = 44;
  )cpp"));
}

TEST_F(ThinBMIDeclsHashTest, OrderingTests) {
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export int v = 43;
export int a() {
  return 43;
}
  )cpp",
                             R"cpp(
export module a;
export int a() {
  return 43;
}
export int v = 43;
  )cpp"));
}

// Tests that the inerface will change every time we touched, added or delete
// an inline function.
TEST_F(ThinBMIDeclsHashTest, InlineFunctionTests) {
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export inline int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export inline int a() {
  return 44;
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export inline int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export inline short a() {
  return 43;
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export inline int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
export inline int a(int v = 44) {
  return 43;
}
  )cpp"));

  // Note that the following cases **can** be fine if the interface didn't
  // change. But we choose to change the interface according to our current
  // implementation strategies to change the interface every time for every
  // change in inline functions.
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp",
                              R"cpp(
export module a;
inline int inl_a() {
  return 44;
}
export int a() {
  return inl_a();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
inline int inl_a() {
  return 45;
}
export int a() {
  return inl_a();
}
  )cpp",
                              R"cpp(
export module a;
inline int inl_a() {
  return 44;
}
export int a() {
  return inl_a();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
inline int inl_a() {
  return 44;
}
export int a() {
  return inl_a();
}
  )cpp",
                              R"cpp(
export module a;
inline int inl_a() {
  return 44;
}
inline int new_unused_inline() {
  return 43;
}
export int a() {
  return inl_a();
}
  )cpp"));

  /// Testing implicitly inline function.

  // Note that the following case **can** be fine if the interface didn't
  // change. But we choose to change the interface according to our current
  // implementation strategies to change the interface every time for every
  // change in inline functions.
  EXPECT_FALSE(CompareBMIHash(R"cpp(
module;
class A {
public:
  int get() { return 43; }
};
export module a;
export int a() {
  A a;
  return a.get();
}
  )cpp",
                              R"cpp(
module;
class A {
public:
  int get() { return 44; }
};
export module a;
export int a() {
  A a;
  return a.get();
}
  )cpp"));
}

// Tests that the inerface will change every time we touched, added or delete
// an inline variables.
TEST_F(ThinBMIDeclsHashTest, InlineVarTests) {
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export inline int a = 43;
  )cpp",
                              R"cpp(
export module a;
export inline int a = 45;
  )cpp"));

  // Note that the following case **can** be fine if the interface didn't
  // change. But we choose to change the interface according to our current
  // implementation strategies to change the interface every time for every
  // change in inline functions.
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
inline int a = 43;
  )cpp",
                              R"cpp(
export module a;
inline int a = 45;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export int a = 43;
  )cpp",
                              R"cpp(
export module a;
export int a = 43;
inline int v = 49;
  )cpp"));
}

TEST_F(ThinBMIDeclsHashTest, ClassTests) {
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
export class A {
public:
  int get() { return 43; }
};
  )cpp",
                             R"cpp(
export module a;
export class A {
public:
  int get() { return 44; }
};
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export class A {
public:
  int get() { return 43; }
};
  )cpp",
                              R"cpp(
export module a;
export class A {
public:
  int get() { return 43; }

  int get_2() { return 44;}
};
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export class A {
public:
  int get() { return 43; }
};
  )cpp",
                              R"cpp(
export module a;
export class A {
  int a = 43;
public:
  int get() { return a; }
};
  )cpp"));

  /// Testing the use of different classes in inline functions.
  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
class A {
public:
  int get() { return 43; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp",
                             R"cpp(
export module a;
class A {
public:
  int get() { return 44; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp"));

  EXPECT_TRUE(CompareBMIHash(R"cpp(
export module a;
class A {
public:
  int get() { return 43; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp",
                             R"cpp(
export module a;
class A {
public:
  int get() { return 43; }
  int get_2() { return 44; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class A {
public:
  int get() { return 43; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp",
                              R"cpp(
export module a;
class A {
  int a = 43;
public:
  int get() { return a; }
};
export inline int func() {
  A a;
  return a.get();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class A {
public:
  int get() { return 43; }
};
export inline A func() {
  return A();
}
  )cpp",
                              R"cpp(
export module a;
class A {
  int a = 43;
public:
  int get() { return a; }
};
export inline A func() {
  return A();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class A {
  int a = 44;
public:
  int get() { return a; }
};
export inline A func() {
  return A();
}
  )cpp",
                              R"cpp(
export module a;
class A {
  int a = 43;
public:
  int get() { return a; }
};
export inline A func() {
  return A();
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class A {
  int a = 44;
public:
  int get() { return a; }
};
export inline A func() {
  return A();
}
  )cpp",
                              R"cpp(
export module a;
class A {
  short a = 44;
public:
  int get() { return a; }
};
export inline A func() {
  return A();
}
  )cpp"));

  // Testing different bases
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export class A {};
  )cpp",
                              R"cpp(
export module a;
class Base1 {};
export class A : public Base1 {};
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class Base1 {};
export class A : public Base1 {};
  )cpp",
                              R"cpp(
export module a;
class Base1 {};
export class A : protected Base1 {};
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class Base1 {};
class Base2 {};
export class A : public Base1 {};
  )cpp",
                              R"cpp(
export module a;
class Base1 {};
class Base2 {};
export class A : public Base1, public Base2 {};
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
class Base1 {};
class Base2 {};
export class A : public Base1, public Base2 {};
  )cpp",
                              R"cpp(
export module a;
class Base1 {};
class Base2 {};
export class A : virtual public Base1, virtual public Base2 {};
  )cpp"));
}

TEST_F(ThinBMIDeclsHashTest, ExportUsingTests) {
  EXPECT_TRUE(CompareBMIHash(R"cpp(
module;
int a() { return 43; }
export module a;
export using ::a;
  )cpp",
                             R"cpp(
module;
int a() { return 44; }
export module a;
export using ::a;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
module;
inline int a() { return 43; }
export module a;
export using ::a;
  )cpp",
                              R"cpp(
module;
inline int a() { return 44; }
export module a;
export using ::a;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
module;
class A {
public:
  int get() { return 43; }
};
export module a;
export using ::A;
  )cpp",
                              R"cpp(
module;
class A {
public:
  int get() { return 44; }
};
export module a;
export using ::A;
  )cpp"));
}

TEST_F(ThinBMIDeclsHashTest, ConstExprTests) {
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export constexpr int a = 43;
  )cpp",
                              R"cpp(
export module a;
export constexpr int a = 44;
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export constexpr int a() {
  return 44;
}
  )cpp",
                              R"cpp(
export module a;
export constexpr int a() {
  return 45;
}
  )cpp"));
}

TEST_F(ThinBMIDeclsHashTest, TemplateTests) {
  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export template <class C>
int get(C c) { return 43 + c; }
  )cpp",
                              R"cpp(
export module a;
export template <class C>
int get(C c) { return 44 + c; }
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
template <class C>
int get(C c) { return 43 + c; }
export int a() {
  return get<int>(43);
}
  )cpp",
                              R"cpp(
export module a;
template <class C>
int get(C c) { return 44 + c; }
export int a() {
  return get<int>(43);
}
  )cpp"));

  EXPECT_FALSE(CompareBMIHash(R"cpp(
export module a;
export template<class T>
class Templ {
public:
  // get is a non-template function inside a template class.
  int get() { return 43; }
};
  )cpp",
                              R"cpp(
export module a;
export template<class T>
class Templ {
public:
  int get() { return 44; }
};
  )cpp"));
}

static std::string get_module_map_flag(StringRef ModuleName,
                                       StringRef BMIPath) {
  return std::string("-fmodule-file=") + ModuleName.str() + "=" + BMIPath.str();
}

// Test that if the decls hash of any import module changes, change
// the decls hash of the current module.
TEST_F(ThinBMIDeclsHashTest, ImportTests) {
  auto BMIPathA = GenerateModuleInterface(R"cpp(
export module a;
export int a() {
  return 43;
}
  )cpp");

  auto BMIPathB = GenerateModuleInterface(R"cpp(
export module b;
import a;
  )cpp",
                                          get_module_map_flag("a", BMIPathA));

  std::optional<uint64_t> Hash1 = getBMIHash(BMIPathB);
  EXPECT_TRUE(Hash1);

  // Test that if the decls hash of the imported doesn't change,
  // the decls hash of the current module shouldn't change.
  BMIPathA = GenerateModuleInterface(R"cpp(
export module a;
export int a() {
  return 44;
}
  )cpp");

  BMIPathB = GenerateModuleInterface(R"cpp(
export module b;
import a;
  )cpp",
                                     get_module_map_flag("a", BMIPathA));

  std::optional<uint64_t> Hash2 = getBMIHash(BMIPathB);
  EXPECT_TRUE(Hash2);

  EXPECT_EQ(*Hash1, *Hash2);

  // Test that if the decls hash of the imported changes,
  // the decls hash of the current module shouldn't change.
  BMIPathA = GenerateModuleInterface(R"cpp(
export module a;
export long long a(int x = 43) {
  return 43;
}
  )cpp");

  BMIPathB = GenerateModuleInterface(R"cpp(
export module b;
import a;
  )cpp",
                                     get_module_map_flag("a", BMIPathA));

  std::optional<uint64_t> Hash3 = getBMIHash(BMIPathB);
  EXPECT_TRUE(Hash2);

  EXPECT_NE(*Hash1, *Hash3);
}

} // namespace
