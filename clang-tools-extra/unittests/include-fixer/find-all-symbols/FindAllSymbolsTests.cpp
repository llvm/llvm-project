//===-- FindAllSymbolsTests.cpp - find all symbols unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbols.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/YAMLTraits.h"
#include "gtest/gtest.h"

namespace clang {
namespace find_all_symbols {

static const char HeaderName[] = "symbols.h";

class MockReporter
    : public clang::find_all_symbols::FindAllSymbols::ResultReporter {
public:
  ~MockReporter() {}

  void reportResult(llvm::StringRef FileName,
                    const SymbolInfo &Symbol) override {
    Symbols.push_back(Symbol);
  }

  bool hasSymbol(const SymbolInfo &Symbol) {
    for (const auto &S : Symbols) {
      if (S == Symbol)
        return true;
    }
    return false;
  }

  bool getSymbolExtraInfo(SymbolInfo *Symbol) {
    for (const auto &S : Symbols) {
      if (S == *Symbol) {
        Symbol->FunctionInfos = S.FunctionInfos;
        Symbol->TypedefNameInfos = S.TypedefNameInfos;
        Symbol->VariableInfos = S.VariableInfos;
        return true;
      }
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

  bool getSymbolExtraInfo(SymbolInfo &Symbol) {
    return Reporter.getSymbolExtraInfo(&Symbol);
  }

  bool runFindAllSymbols(StringRef Code) {
    FindAllSymbols matcher(&Reporter);
    clang::ast_matchers::MatchFinder MatchFinder;
    matcher.registerMatchers(&MatchFinder);

    llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
        new vfs::InMemoryFileSystem);
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    std::string FileName = "symbol.cc";
    std::unique_ptr<clang::tooling::FrontendActionFactory> Factory =
        clang::tooling::newFrontendActionFactory(&MatchFinder);
    tooling::ToolInvocation Invocation(
        {std::string("find_all_symbols"), std::string("-fsyntax-only"),
         FileName},
        Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    InMemoryFileSystem->addFile(HeaderName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Code));

    std::string Content = "#include\"" + std::string(HeaderName) + "\"";
    InMemoryFileSystem->addFile(FileName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Content));
    Invocation.run();
    return true;
  }

private:
  MockReporter Reporter;
};

SymbolInfo
CreateSymbolInfo(StringRef Name, SymbolInfo::SymbolKind Type,
                 const std::string FilePath, int LineNumber,
                 const std::vector<SymbolInfo::Context> &Contexts) {
  SymbolInfo Symbol;
  Symbol.Name = Name;
  Symbol.Type = Type;
  Symbol.FilePath = FilePath;
  Symbol.LineNumber = LineNumber;
  Symbol.Contexts = Contexts;
  return Symbol;
}

TEST_F(FindAllSymbolsTest, VariableSymbols) {
  static const char Code[] = R"(
      extern int xargc;
      namespace na {
      static bool SSSS = false;
      namespace nb { const long long *XXXX; }
      })";
  runFindAllSymbols(Code);

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("xargc", SymbolInfo::Variable, HeaderName, 2, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.VariableInfos.getValue().Type);
  }
  {
    SymbolInfo Symbol =
        CreateSymbolInfo("SSSS", SymbolInfo::Variable, HeaderName, 4,
                         {{SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("_Bool", Symbol.VariableInfos.getValue().Type);
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo(
        "XXXX", SymbolInfo::Variable, HeaderName, 5,
        {{SymbolInfo::Namespace, "nb"}, {SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("const long long *", Symbol.VariableInfos.getValue().Type);
  }
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

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("C_Func", SymbolInfo::Function, HeaderName, 3, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.FunctionInfos.getValue().ReturnType);
    EXPECT_TRUE(Symbol.FunctionInfos.getValue().ParameterTypes.empty());
  }
  {
    SymbolInfo Symbol =
        CreateSymbolInfo("C_struct", SymbolInfo::Class, HeaderName, 4, {});
    EXPECT_TRUE(hasSymbol(Symbol));
  }
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

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("Glob", SymbolInfo::Class, HeaderName, 2, {});
    EXPECT_TRUE(hasSymbol(Symbol));
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo("A", SymbolInfo::Class, HeaderName, 6,
                                         {{SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
  }
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

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("T_TEMP", SymbolInfo::Class, HeaderName, 3, {});
    EXPECT_TRUE(hasSymbol(Symbol));
  }
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

  {
    SymbolInfo Symbol = CreateSymbolInfo("gg", SymbolInfo::Class, HeaderName, 3,
                                         {{SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.FunctionInfos.getValue().ReturnType);
    EXPECT_EQ(1u, Symbol.FunctionInfos.getValue().ParameterTypes.size());
    EXPECT_EQ("int", Symbol.FunctionInfos.getValue().ParameterTypes[0]);
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo("f", SymbolInfo::Class, HeaderName, 4,
                                         {{SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.FunctionInfos.getValue().ReturnType);
    EXPECT_EQ(1u, Symbol.FunctionInfos.getValue().ParameterTypes.size());
    EXPECT_EQ("const int &", Symbol.FunctionInfos.getValue().ParameterTypes[0]);
  }
  {
    SymbolInfo Symbol =
        CreateSymbolInfo("SSSFFF", SymbolInfo::Class, HeaderName, 5,
                         {{SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("void", Symbol.FunctionInfos.getValue().ReturnType);
    EXPECT_TRUE(Symbol.FunctionInfos.getValue().ParameterTypes.empty());
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo(
        "fun", SymbolInfo::Class, HeaderName, 10,
        {{SymbolInfo::Namespace, "nb"}, {SymbolInfo::Namespace, "na"}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("void", Symbol.FunctionInfos.getValue().ReturnType);
    EXPECT_EQ(1u, Symbol.FunctionInfos.getValue().ParameterTypes.size());
    EXPECT_EQ("T", Symbol.FunctionInfos.getValue().ParameterTypes[0]);
  }
}

TEST_F(FindAllSymbolsTest, NamespaceTest) {
  static const char Code[] = R"(
      int X1;
      namespace { int X2; }
      namespace { namespace { int X3; } }
      namespace { namespace nb { int X4;} }
      )";
  runFindAllSymbols(Code);

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("X1", SymbolInfo::Variable, HeaderName, 2, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.VariableInfos.getValue().Type);
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo("X2", SymbolInfo::Variable, HeaderName,
                                         3, {{SymbolInfo::Namespace, ""}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.VariableInfos.getValue().Type);
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo(
        "X3", SymbolInfo::Variable, HeaderName, 4,
        {{SymbolInfo::Namespace, ""}, {SymbolInfo::Namespace, ""}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.VariableInfos.getValue().Type);
  }
  {
    SymbolInfo Symbol = CreateSymbolInfo(
        "X4", SymbolInfo::Variable, HeaderName, 5,
        {{SymbolInfo::Namespace, "nb"}, {SymbolInfo::Namespace, ""}});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("int", Symbol.VariableInfos.getValue().Type);
  }
}

TEST_F(FindAllSymbolsTest, DecayedTypeTest) {
  static const char Code[] = "void DecayedFunc(int x[], int y[10]) {}";
  runFindAllSymbols(Code);
  SymbolInfo Symbol =
      CreateSymbolInfo("DecayedFunc", SymbolInfo::Class, HeaderName, 1, {});
  EXPECT_TRUE(hasSymbol(Symbol));
  getSymbolExtraInfo(Symbol);
  EXPECT_EQ("void", Symbol.FunctionInfos.getValue().ReturnType);
  EXPECT_EQ(2u, Symbol.FunctionInfos.getValue().ParameterTypes.size());
  EXPECT_EQ("int *", Symbol.FunctionInfos.getValue().ParameterTypes[0]);
  EXPECT_EQ("int *", Symbol.FunctionInfos.getValue().ParameterTypes[1]);
}

TEST_F(FindAllSymbolsTest, CTypedefTest) {
  static const char Code[] = R"(
      typedef unsigned size_t_;
      typedef struct { int x; } X;
      using XX = X;
      )";
  runFindAllSymbols(Code);

  {
    SymbolInfo Symbol =
        CreateSymbolInfo("size_t_", SymbolInfo::TypedefName, HeaderName, 2, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("unsigned int",
              Symbol.TypedefNameInfos.getValue().UnderlyingType);
  }
  {
    SymbolInfo Symbol =
        CreateSymbolInfo("X", SymbolInfo::TypedefName, HeaderName, 3, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("struct X", Symbol.TypedefNameInfos.getValue().UnderlyingType);
  }
  {
    SymbolInfo Symbol =
        CreateSymbolInfo("XX", SymbolInfo::TypedefName, HeaderName, 4, {});
    EXPECT_TRUE(hasSymbol(Symbol));
    getSymbolExtraInfo(Symbol);
    EXPECT_EQ("X", Symbol.TypedefNameInfos.getValue().UnderlyingType);
  }
}

} // namespace find_all_symbols
} // namespace clang
