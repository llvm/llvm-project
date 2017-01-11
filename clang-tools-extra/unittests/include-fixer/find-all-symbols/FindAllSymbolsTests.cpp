//===-- FindAllSymbolsTests.cpp - find all symbols unit tests ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbolsAction.h"
#include "HeaderMapCollector.h"
#include "SymbolInfo.h"
#include "SymbolReporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace find_all_symbols {

static const char HeaderName[] = "symbols.h";

class TestSymbolReporter : public clang::find_all_symbols::SymbolReporter {
public:
  ~TestSymbolReporter() override {}

  void reportSymbol(llvm::StringRef FileName,
                    const SymbolInfo &Symbol) override {
    Symbols.push_back(Symbol);
  }

  bool hasSymbol(const SymbolInfo &Symbol) const {
    for (const auto &S : Symbols) {
      if (S == Symbol)
        return true;
    }
    return false;
  }

private:
  std::vector<SymbolInfo> Symbols;
};

class FindAllSymbolsTest : public ::testing::Test {
public:
  bool hasSymbol(const SymbolInfo &Symbol) {
    return Reporter.hasSymbol(Symbol);
  }

  bool runFindAllSymbols(StringRef Code) {
    llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
        new vfs::InMemoryFileSystem);
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    std::string FileName = "symbol.cc";

    const std::string InternalHeader = "internal/internal_header.h";
    const std::string TopHeader = "<top>";
    // Test .inc header path. The header for `IncHeaderClass` should be
    // internal.h, which will eventually be mapped to <top>.
    std::string IncHeader = "internal/private.inc";
    std::string IncHeaderCode = "class IncHeaderClass {};";

    HeaderMapCollector::RegexHeaderMap RegexMap = {
        {R"(internal_.*\.h$)", TopHeader.c_str()},
    };

    std::string InternalCode =
        "#include \"private.inc\"\nclass Internal {};";
    SymbolInfo InternalSymbol("Internal", SymbolInfo::SymbolKind::Class,
                              TopHeader, 2, {});
    SymbolInfo IncSymbol("IncHeaderClass", SymbolInfo::SymbolKind::Class,
                         TopHeader, 1, {});
    InMemoryFileSystem->addFile(
        IncHeader, 0, llvm::MemoryBuffer::getMemBuffer(IncHeaderCode));
    InMemoryFileSystem->addFile(InternalHeader, 0,
                                llvm::MemoryBuffer::getMemBuffer(InternalCode));

    std::unique_ptr<clang::tooling::FrontendActionFactory> Factory(
        new FindAllSymbolsActionFactory(&Reporter, &RegexMap));

    tooling::ToolInvocation Invocation(
        {std::string("find_all_symbols"), std::string("-fsyntax-only"),
         std::string("-std=c++11"), FileName},
        Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    InMemoryFileSystem->addFile(HeaderName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Code));

    std::string Content = "#include\"" + std::string(HeaderName) +
                          "\"\n"
                          "#include \"" +
                          InternalHeader + "\"";
#if !defined(_MSC_VER) && !defined(__MINGW32__)
    // Test path cleaning for both decls and macros.
    const std::string DirtyHeader = "./internal/./a/b.h";
    Content += "\n#include \"" + DirtyHeader + "\"";
    const std::string CleanHeader = "internal/a/b.h";
    const std::string DirtyHeaderContent =
        "#define INTERNAL 1\nclass ExtraInternal {};";
    InMemoryFileSystem->addFile(
        DirtyHeader, 0, llvm::MemoryBuffer::getMemBuffer(DirtyHeaderContent));
    SymbolInfo DirtyMacro("INTERNAL", SymbolInfo::SymbolKind::Macro,
                          CleanHeader, 1, {});
    SymbolInfo DirtySymbol("ExtraInternal", SymbolInfo::SymbolKind::Class,
                           CleanHeader, 2, {});
#endif // _MSC_VER && __MINGW32__
    InMemoryFileSystem->addFile(FileName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Content));
    Invocation.run();
    EXPECT_TRUE(hasSymbol(InternalSymbol));
    EXPECT_TRUE(hasSymbol(IncSymbol));
#if !defined(_MSC_VER) && !defined(__MINGW32__)
    EXPECT_TRUE(hasSymbol(DirtySymbol));
    EXPECT_TRUE(hasSymbol(DirtyMacro));
#endif  // _MSC_VER && __MINGW32__
    return true;
  }

protected:
  TestSymbolReporter Reporter;
};

TEST_F(FindAllSymbolsTest, VariableSymbols) {
  static const char Code[] = R"(
      extern int xargc;
      namespace na {
      static bool SSSS = false;
      namespace nb { const long long *XXXX; }
      })";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("xargc", SymbolInfo::SymbolKind::Variable, HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("SSSS", SymbolInfo::SymbolKind::Variable, HeaderName, 4,
                      {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("XXXX", SymbolInfo::SymbolKind::Variable, HeaderName, 5,
                      {{SymbolInfo::ContextType::Namespace, "nb"},
                       {SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, ExternCSymbols) {
  static const char Code[] = R"(
      extern "C" {
      int C_Func() { return 0; }
      struct C_struct {
        int Member;
      };
      })";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("C_Func", SymbolInfo::SymbolKind::Function, HeaderName, 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("C_struct", SymbolInfo::SymbolKind::Class, HeaderName, 4, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, CXXRecordSymbols) {
  static const char Code[] = R"(
      struct Glob {};
      struct A; // Not a defintion, ignored.
      class NOP; // Not a defintion, ignored
      namespace na {
      struct A {
        struct AAAA {};
        int x;
        int y;
        void f() {}
      };
      };  //
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("Glob", SymbolInfo::SymbolKind::Class, HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("A", SymbolInfo::SymbolKind::Class, HeaderName, 6,
                      {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("AAA", SymbolInfo::SymbolKind::Class, HeaderName, 7,
                      {{SymbolInfo::ContextType::Record, "A"},
                       {SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_FALSE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, CXXRecordSymbolsTemplate) {
  static const char Code[] = R"(
      template <typename T>
      class T_TEMP {
        template <typename _Tp1>
        struct rebind { typedef T_TEMP<_Tp1> other; };
      };
      // Ignore specialization.
      template class T_TEMP<char>;

      template <typename T>
      class Observer {
      };
      // Ignore specialization.
      template <> class Observer<int> {};
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("T_TEMP", SymbolInfo::SymbolKind::Class, HeaderName, 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, DontIgnoreTemplatePartialSpecialization) {
  static const char Code[] = R"(
      template<class> class Class; // undefined
      template<class R, class... ArgTypes>
      class Class<R(ArgTypes...)> {
      };

      template<class T> void f() {};
      template<> void f<int>() {};
      )";
  runFindAllSymbols(Code);
  SymbolInfo Symbol =
      SymbolInfo("Class", SymbolInfo::SymbolKind::Class, HeaderName, 4, {});
  EXPECT_TRUE(hasSymbol(Symbol));
  Symbol =
      SymbolInfo("f", SymbolInfo::SymbolKind::Function, HeaderName, 7, {});
  EXPECT_TRUE(hasSymbol(Symbol));
  Symbol =
      SymbolInfo("f", SymbolInfo::SymbolKind::Function, HeaderName, 8, {});
  EXPECT_FALSE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, FunctionSymbols) {
  static const char Code[] = R"(
      namespace na {
      int gg(int);
      int f(const int &a) { int Local; static int StaticLocal; return 0; }
      static void SSSFFF() {}
      }  // namespace na
      namespace na {
      namespace nb {
      template<typename T>
      void fun(T t) {};
      } // namespace nb
      } // namespace na";
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("gg", SymbolInfo::SymbolKind::Function, HeaderName, 3,
                 {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("f", SymbolInfo::SymbolKind::Function, HeaderName, 4,
                      {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("SSSFFF", SymbolInfo::SymbolKind::Function, HeaderName, 5,
                      {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("fun", SymbolInfo::SymbolKind::Function, HeaderName, 10,
                      {{SymbolInfo::ContextType::Namespace, "nb"},
                       {SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, NamespaceTest) {
  static const char Code[] = R"(
      int X1;
      namespace { int X2; }
      namespace { namespace { int X3; } }
      namespace { namespace nb { int X4; } }
      namespace na { inline namespace __1 { int X5; } }
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("X1", SymbolInfo::SymbolKind::Variable, HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("X2", SymbolInfo::SymbolKind::Variable, HeaderName, 3,
                      {{SymbolInfo::ContextType::Namespace, ""}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("X3", SymbolInfo::SymbolKind::Variable, HeaderName, 4,
                      {{SymbolInfo::ContextType::Namespace, ""},
                       {SymbolInfo::ContextType::Namespace, ""}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("X4", SymbolInfo::SymbolKind::Variable, HeaderName, 5,
                      {{SymbolInfo::ContextType::Namespace, "nb"},
                       {SymbolInfo::ContextType::Namespace, ""}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("X5", SymbolInfo::SymbolKind::Variable, HeaderName, 6,
                      {{SymbolInfo::ContextType::Namespace, "na"}});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, DecayedTypeTest) {
  static const char Code[] = "void DecayedFunc(int x[], int y[10]) {}";
  runFindAllSymbols(Code);
  SymbolInfo Symbol = SymbolInfo(
      "DecayedFunc", SymbolInfo::SymbolKind::Function, HeaderName, 1, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, CTypedefTest) {
  static const char Code[] = R"(
      typedef unsigned size_t_;
      typedef struct { int x; } X;
      using XX = X;
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol = SymbolInfo("size_t_", SymbolInfo::SymbolKind::TypedefName,
                                 HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("X", SymbolInfo::SymbolKind::TypedefName, HeaderName, 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("XX", SymbolInfo::SymbolKind::TypedefName, HeaderName, 4, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, EnumTest) {
  static const char Code[] = R"(
      enum Glob_E { G1, G2 };
      enum class Altitude { high='h', low='l'};
      enum { A1, A2 };
      class A {
      public:
        enum A_ENUM { X1, X2 };
      };
      enum DECL : int;
      )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("Glob_E", SymbolInfo::SymbolKind::EnumDecl, HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("G1", SymbolInfo::SymbolKind::EnumConstantDecl, HeaderName, 2,
                 {{SymbolInfo::ContextType::EnumDecl, "Glob_E"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("G2", SymbolInfo::SymbolKind::EnumConstantDecl, HeaderName, 2,
                 {{SymbolInfo::ContextType::EnumDecl, "Glob_E"}});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("Altitude", SymbolInfo::SymbolKind::EnumDecl, HeaderName,
                      3, {});
  EXPECT_TRUE(hasSymbol(Symbol));
  Symbol =
      SymbolInfo("high", SymbolInfo::SymbolKind::EnumConstantDecl, HeaderName,
                 3, {{SymbolInfo::ContextType::EnumDecl, "Altitude"}});
  EXPECT_FALSE(hasSymbol(Symbol));

  Symbol = SymbolInfo("A1", SymbolInfo::SymbolKind::EnumConstantDecl,
                      HeaderName, 4, {{SymbolInfo::ContextType::EnumDecl, ""}});
  EXPECT_TRUE(hasSymbol(Symbol));
  Symbol = SymbolInfo("A2", SymbolInfo::SymbolKind::EnumConstantDecl,
                      HeaderName, 4, {{SymbolInfo::ContextType::EnumDecl, ""}});
  EXPECT_TRUE(hasSymbol(Symbol));
  Symbol = SymbolInfo("", SymbolInfo::SymbolKind::EnumDecl, HeaderName, 4, {});
  EXPECT_FALSE(hasSymbol(Symbol));

  Symbol = SymbolInfo("A_ENUM", SymbolInfo::SymbolKind::EnumDecl, HeaderName, 7,
                      {{SymbolInfo::ContextType::Record, "A"}});
  EXPECT_FALSE(hasSymbol(Symbol));

  Symbol = SymbolInfo("X1", SymbolInfo::SymbolKind::EnumDecl, HeaderName, 7,
                      {{SymbolInfo::ContextType::EnumDecl, "A_ENUM"},
                       {SymbolInfo::ContextType::Record, "A"}});
  EXPECT_FALSE(hasSymbol(Symbol));

  Symbol =
      SymbolInfo("DECL", SymbolInfo::SymbolKind::EnumDecl, HeaderName, 9, {});
  EXPECT_FALSE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, IWYUPrivatePragmaTest) {
  static const char Code[] = R"(
    // IWYU pragma: private, include "bar.h"
    struct Bar {
    };
  )";
  runFindAllSymbols(Code);

  SymbolInfo Symbol =
      SymbolInfo("Bar", SymbolInfo::SymbolKind::Class, "bar.h", 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, MacroTest) {
  static const char Code[] = R"(
    #define X
    #define Y 1
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
  )";
  runFindAllSymbols(Code);
  SymbolInfo Symbol =
      SymbolInfo("X", SymbolInfo::SymbolKind::Macro, HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("Y", SymbolInfo::SymbolKind::Macro, HeaderName, 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("MAX", SymbolInfo::SymbolKind::Macro, HeaderName, 4, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, MacroTestWithIWYU) {
  static const char Code[] = R"(
    // IWYU pragma: private, include "bar.h"
    #define X
    #define Y 1
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
  )";
  runFindAllSymbols(Code);
  SymbolInfo Symbol =
      SymbolInfo("X", SymbolInfo::SymbolKind::Macro, "bar.h", 3, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("Y", SymbolInfo::SymbolKind::Macro, "bar.h", 4, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("MAX", SymbolInfo::SymbolKind::Macro, "bar.h", 5, {});
  EXPECT_TRUE(hasSymbol(Symbol));
}

TEST_F(FindAllSymbolsTest, NoFriendTest) {
  static const char Code[] = R"(
    class WorstFriend {
      friend void Friend();
      friend class BestFriend;
    };
  )";
  runFindAllSymbols(Code);
  SymbolInfo Symbol = SymbolInfo("WorstFriend", SymbolInfo::SymbolKind::Class,
                                 HeaderName, 2, {});
  EXPECT_TRUE(hasSymbol(Symbol));

  Symbol = SymbolInfo("Friend", SymbolInfo::SymbolKind::Function, HeaderName,
                      3, {});
  EXPECT_FALSE(hasSymbol(Symbol));

  Symbol = SymbolInfo("BestFriend", SymbolInfo::SymbolKind::Class, HeaderName,
                      4, {});
  EXPECT_FALSE(hasSymbol(Symbol));
}

} // namespace find_all_symbols
} // namespace clang
