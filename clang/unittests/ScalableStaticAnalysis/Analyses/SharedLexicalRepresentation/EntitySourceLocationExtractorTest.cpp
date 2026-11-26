//===- EntitySourceLocationExtractorTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Analyses/SharedLexicalRepresentation/SharedLexicalRepresentation.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace clang::ssaf {

extern llvm::ArrayRef<SourceLocationRecord>
getDeclLocations(const EntitySourceLocationsSummary &);

namespace {

class EntitySourceLocationExtractorTest : public TestFixture {
protected:
  using PathString = llvm::SmallString<128>;

  PathString TestDir;
  PathString SourceFile;
  SSAFOptions Opts;
  TUSummary TUSum;
  TUSummaryBuilder Builder;
  std::unique_ptr<TUSummaryExtractor> Extractor;
  std::unique_ptr<ASTUnit> AST;

  EntitySourceLocationExtractorTest()
      : TUSum(llvm::Triple("arm64-apple-macosx"),
              BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSum, Opts) {}

  void SetUp() override {
    std::error_code EC =
        llvm::sys::fs::createUniqueDirectory("esl-extractor-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

  PathString makePath(llvm::StringRef RelPath) const {
    PathString P = TestDir;
    llvm::sys::path::append(P, RelPath);
    return P;
  }

  PathString realPathOf(llvm::StringRef AbsPath) const {
    PathString Real;
    std::error_code EC = llvm::sys::fs::real_path(AbsPath, Real);
    EXPECT_FALSE(EC) << "real_path failed for " << AbsPath.str() << ": "
                     << EC.message();
    return Real;
  }

  void writeFile(llvm::StringRef AbsPath, llvm::StringRef Content) const {
    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC) << "Failed to open " << AbsPath.str() << ": "
                     << EC.message();
    OS << Content;
  }

  void runExtractor(llvm::StringRef SourceAbsPath, llvm::StringRef Code,
                    std::vector<std::string> ExtraArgs = {}) {
    SourceFile = SourceAbsPath;
    AST = tooling::buildASTFromCodeWithArgs(Code, ExtraArgs, SourceAbsPath,
                                            "clang-tool");
    ASSERT_TRUE(AST) << "Failed to build AST";

    for (auto &E : TUSummaryExtractorRegistry::entries()) {
      if (E.getName() == EntitySourceLocationsSummary::Name) {
        Extractor = E.instantiate(Builder);
        break;
      }
    }
    ASSERT_TRUE(Extractor) << "EntitySourceLocationExtractor not registered";
    Extractor->HandleTranslationUnit(AST->getASTContext());
  }

  void setUpFromCode(llvm::StringRef Code,
                     llvm::StringRef SourceName = "test.cpp",
                     std::vector<std::string> ExtraArgs = {}) {
    PathString AbsPath = makePath(SourceName);
    writeFile(AbsPath, Code);
    runExtractor(AbsPath, Code, std::move(ExtraArgs));
  }

  template <typename Pred> const NamedDecl *findDecl(Pred P) const {
    class Finder : public DynamicRecursiveASTVisitor {
    public:
      Pred &Predicate;
      const NamedDecl *Found = nullptr;
      explicit Finder(Pred &P) : Predicate(P) {
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }
      bool VisitNamedDecl(NamedDecl *D) override {
        if (D && Predicate(D)) {
          Found = D;
          return false;
        }
        return true;
      }
    };
    Finder F(P);
    F.TraverseAST(AST->getASTContext());
    return F.Found;
  }

  const NamedDecl *findDeclByName(llvm::StringRef Name) const {
    return findDecl(
        [&](const NamedDecl *D) { return D->getNameAsString() == Name.str(); });
  }

  std::vector<const FunctionDecl *>
  findAllFunctions(llvm::StringRef Name) const {
    class Finder : public DynamicRecursiveASTVisitor {
    public:
      std::string Name;
      std::vector<const FunctionDecl *> Found;
      explicit Finder(llvm::StringRef N) : Name(N.str()) {
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }
      bool VisitFunctionDecl(FunctionDecl *FD) override {
        if (FD && FD->getNameAsString() == Name)
          Found.push_back(FD);
        return true;
      }
    };
    Finder F(Name);
    F.TraverseAST(AST->getASTContext());
    return F.Found;
  }

  llvm::ArrayRef<SourceLocationRecord> recordsFor(EntityId Id) const {
    auto &Data = getData(TUSum);
    auto It = Data.find(EntitySourceLocationsSummary::summaryName());
    if (It == Data.end())
      return {};
    auto It2 = It->second.find(Id);
    if (It2 == It->second.end())
      return {};
    return getDeclLocations(
        *static_cast<const EntitySourceLocationsSummary *>(It2->second.get()));
  }

  llvm::ArrayRef<SourceLocationRecord>
  recordsForDecl(const NamedDecl *D) const {
    if (!D)
      return {};
    auto Id = Extractor->addEntity(D);
    if (!Id)
      return {};
    return recordsFor(*Id);
  }

  llvm::ArrayRef<SourceLocationRecord>
  recordsForReturn(const FunctionDecl *FD) const {
    if (!FD)
      return {};
    auto Id = Extractor->addEntityForReturn(FD);
    if (!Id)
      return {};
    return recordsFor(*Id);
  }
};

TEST_F(EntitySourceLocationExtractorTest, IsExtractorRegistered) {
  EXPECT_TRUE(isTUSummaryExtractorRegistered("EntitySourceLocations"));
}

TEST_F(EntitySourceLocationExtractorTest, SingleDecl) {
  setUpFromCode("int *p;\n");

  const auto *D = findDeclByName("p");
  ASSERT_NE(D, nullptr);
  auto Recs = recordsForDecl(D);
  ASSERT_EQ(Recs.size(), 1u);
  EXPECT_EQ(Recs[0].Line, 1u);
  EXPECT_EQ(Recs[0].Column, 6u);
  EXPECT_EQ(Recs[0].FilePath, realPathOf(SourceFile).str());
}

TEST_F(EntitySourceLocationExtractorTest, MultipleRedecls) {
  setUpFromCode("extern int p;\n"
                "int p = 0;\n");

  const auto *D = findDeclByName("p");
  ASSERT_NE(D, nullptr);
  auto Recs = recordsForDecl(D);
  ASSERT_EQ(Recs.size(), 2u);

  EXPECT_EQ(Recs[0].FilePath, Recs[1].FilePath);
  std::vector<std::pair<unsigned, unsigned>> LinesCols{
      {Recs[0].Line, Recs[0].Column}, {Recs[1].Line, Recs[1].Column}};
  llvm::sort(LinesCols);
  EXPECT_EQ(LinesCols,
            (std::vector<std::pair<unsigned, unsigned>>{{1u, 12u}, {2u, 5u}}));
}

TEST_F(EntitySourceLocationExtractorTest, MultiDeclarator) {
  setUpFromCode("int *p, *q;\n");

  const auto *DP = findDeclByName("p");
  const auto *DQ = findDeclByName("q");
  ASSERT_NE(DP, nullptr);
  ASSERT_NE(DQ, nullptr);

  auto RecP = recordsForDecl(DP);
  auto RecQ = recordsForDecl(DQ);
  ASSERT_EQ(RecP.size(), 1u);
  ASSERT_EQ(RecQ.size(), 1u);

  EXPECT_EQ(RecP[0].Line, RecQ[0].Line);
  EXPECT_EQ(RecP[0].FilePath, RecQ[0].FilePath);
  EXPECT_LT(RecP[0].Column, RecQ[0].Column);
}

TEST_F(EntitySourceLocationExtractorTest, HeaderIncluded) {
  PathString HeaderPath = makePath("shared.h");
  writeFile(HeaderPath, "static int x;\n");
  PathString SrcPath = makePath("test.cpp");
  std::string Code = "#include \"shared.h\"\n";
  writeFile(SrcPath, Code);
  runExtractor(SrcPath, Code, {("-I" + TestDir.str()).str()});

  const auto *D = findDeclByName("x");
  ASSERT_NE(D, nullptr);
  auto Recs = recordsForDecl(D);
  ASSERT_EQ(Recs.size(), 1u);
  EXPECT_EQ(Recs[0].FilePath, realPathOf(HeaderPath).str());
  EXPECT_EQ(Recs[0].Line, 1u);
}

TEST_F(EntitySourceLocationExtractorTest, RealPathSymlink) {
#ifdef _WIN32
  GTEST_SKIP() << "Symlink semantics differ on Windows.";
#endif
  PathString RealHeader = makePath("real-shared.h");
  writeFile(RealHeader, "static int y;\n");
  PathString SymlinkHeader = makePath("via-symlink.h");
  std::error_code EC = llvm::sys::fs::create_link(RealHeader, SymlinkHeader);
  if (EC)
    GTEST_SKIP() << "Failed to create symlink: " << EC.message();

  PathString SrcPath = makePath("symlink-test.cpp");
  std::string Code = "#include \"via-symlink.h\"\n";
  writeFile(SrcPath, Code);
  runExtractor(SrcPath, Code, {("-I" + TestDir.str()).str()});

  const auto *D = findDeclByName("y");
  ASSERT_NE(D, nullptr);
  auto Recs = recordsForDecl(D);
  ASSERT_EQ(Recs.size(), 1u);
  EXPECT_EQ(Recs[0].FilePath, realPathOf(RealHeader).str());
}

TEST_F(EntitySourceLocationExtractorTest, ReturnSlotRecord) {
  setUpFromCode("int *foo();\n");

  auto Funcs = findAllFunctions("foo");
  ASSERT_EQ(Funcs.size(), 1u);
  const FunctionDecl *FD = Funcs[0];

  auto FuncRecs = recordsForDecl(FD);
  ASSERT_EQ(FuncRecs.size(), 1u);
  EXPECT_EQ(FuncRecs[0].Line, 1u);
  EXPECT_EQ(FuncRecs[0].Column, 6u);

  auto RetRecs = recordsForReturn(FD);
  ASSERT_EQ(RetRecs.size(), 1u);
  EXPECT_EQ(RetRecs[0].Line, 1u);
  EXPECT_EQ(RetRecs[0].Column, 1u);
}

TEST_F(EntitySourceLocationExtractorTest, NamedParamTypeSpecStart) {
  setUpFromCode("void foo(int *p);\n");

  auto Funcs = findAllFunctions("foo");
  ASSERT_EQ(Funcs.size(), 1u);
  const FunctionDecl *FD = Funcs[0];
  ASSERT_EQ(FD->getNumParams(), 1u);
  const ParmVarDecl *P = FD->getParamDecl(0);

  auto Recs = recordsForDecl(P);
  ASSERT_EQ(Recs.size(), 1u);
  EXPECT_EQ(Recs[0].Line, 1u);
  EXPECT_EQ(Recs[0].Column, 10u);
}

TEST_F(EntitySourceLocationExtractorTest, UnnamedParamTypeSpecStart) {
  setUpFromCode("void foo(int *);\n");

  auto Funcs = findAllFunctions("foo");
  ASSERT_EQ(Funcs.size(), 1u);
  const FunctionDecl *FD = Funcs[0];
  ASSERT_EQ(FD->getNumParams(), 1u);
  const ParmVarDecl *P = FD->getParamDecl(0);

  auto Recs = recordsForDecl(P);
  ASSERT_EQ(Recs.size(), 1u);
  EXPECT_EQ(Recs[0].Line, 1u);
  EXPECT_EQ(Recs[0].Column, 10u);
}

TEST_F(EntitySourceLocationExtractorTest, FunctionMultiRedecl) {
  setUpFromCode("int *foo(int *p);\n"
                "\n"
                "int *foo(int *p) { return p; }\n");

  auto Funcs = findAllFunctions("foo");
  ASSERT_EQ(Funcs.size(), 2u);
  const FunctionDecl *FD0 = Funcs[0];
  const FunctionDecl *FD1 = Funcs[1];
  ASSERT_EQ(FD0->getNumParams(), 1u);

  auto FuncRecs = recordsForDecl(FD0);
  EXPECT_EQ(FuncRecs.size(), 2u);

  auto RetRecs = recordsForReturn(FD0);
  EXPECT_EQ(RetRecs.size(), 2u);

  auto P0Id = Extractor->addEntity(FD0->getParamDecl(0));
  auto P1Id = Extractor->addEntity(FD1->getParamDecl(0));
  ASSERT_TRUE(P0Id.has_value());
  ASSERT_TRUE(P1Id.has_value());
  EXPECT_EQ(*P0Id, *P1Id);
  auto ParmRecs = recordsFor(*P0Id);
  EXPECT_EQ(ParmRecs.size(), 2u);

  std::vector<unsigned> FuncLines{FuncRecs[0].Line, FuncRecs[1].Line};
  llvm::sort(FuncLines);
  EXPECT_EQ(FuncLines, (std::vector<unsigned>{1u, 3u}));
}

TEST_F(EntitySourceLocationExtractorTest, TemplateInstantiationsShareLoc) {
  setUpFromCode("template <class T> void f(T *p) {}\n"
                "void use() {\n"
                "  f<int>(nullptr);\n"
                "  f<long>(nullptr);\n"
                "}\n");

  auto Funcs = findAllFunctions("f");
  ASSERT_GE(Funcs.size(), 3u);

  std::set<EntityId> DistinctParmIds;
  std::vector<SourceLocationRecord> ParamLocs;
  for (const FunctionDecl *FD : Funcs) {
    if (FD->getNumParams() != 1u)
      continue;
    auto Id = Extractor->addEntity(FD->getParamDecl(0));
    if (!Id)
      continue;
    DistinctParmIds.insert(*Id);
    auto Recs = recordsFor(*Id);
    if (Recs.empty())
      continue;
    ParamLocs.push_back(Recs.front());
  }
  ASSERT_GE(DistinctParmIds.size(), 2u);
  ASSERT_GE(ParamLocs.size(), 2u);

  const SourceLocationRecord &Anchor = ParamLocs.front();
  for (const auto &R : ParamLocs)
    EXPECT_EQ(R, Anchor);
}

TEST_F(EntitySourceLocationExtractorTest, ImplicitDeclSkipped) {
  setUpFromCode("struct S { int *p; };\n"
                "void use() { S s; (void)s; }\n");

  const auto *DS = findDeclByName("S");
  const auto *DP = findDeclByName("p");
  ASSERT_NE(DS, nullptr);
  ASSERT_NE(DP, nullptr);
  EXPECT_FALSE(recordsForDecl(DS).empty());
  EXPECT_FALSE(recordsForDecl(DP).empty());

  class ImplicitMethodFinder : public DynamicRecursiveASTVisitor {
  public:
    std::vector<const CXXMethodDecl *> ImplicitMethods;
    ImplicitMethodFinder() {
      ShouldVisitTemplateInstantiations = true;
      ShouldVisitImplicitCode = true;
    }
    bool VisitCXXMethodDecl(CXXMethodDecl *MD) override {
      if (MD && MD->isImplicit() && MD->getParent() &&
          MD->getParent()->getNameAsString() == "S")
        ImplicitMethods.push_back(MD);
      return true;
    }
  };
  ImplicitMethodFinder F;
  F.TraverseAST(AST->getASTContext());

  ASSERT_FALSE(F.ImplicitMethods.empty());
  for (const CXXMethodDecl *MD : F.ImplicitMethods) {
    EXPECT_FALSE(Extractor->addEntity(MD).has_value());
    EXPECT_TRUE(recordsForDecl(MD).empty());
  }
}

TEST_F(EntitySourceLocationExtractorTest, VariadicTerminatorNotEmitted) {
  setUpFromCode("int my_printf(const char *fmt, ...);\n");

  auto Funcs = findAllFunctions("my_printf");
  ASSERT_EQ(Funcs.size(), 1u);
  const FunctionDecl *FD = Funcs[0];
  ASSERT_EQ(FD->getNumParams(), 1u);
  EXPECT_TRUE(FD->isVariadic());

  EXPECT_FALSE(recordsForDecl(FD).empty());
  EXPECT_FALSE(recordsForReturn(FD).empty());
  EXPECT_FALSE(recordsForDecl(FD->getParamDecl(0)).empty());

  auto FuncId = Extractor->addEntity(FD);
  auto RetId = Extractor->addEntityForReturn(FD);
  auto ParmId = Extractor->addEntity(FD->getParamDecl(0));
  ASSERT_TRUE(FuncId.has_value());
  ASSERT_TRUE(RetId.has_value());
  ASSERT_TRUE(ParmId.has_value());
  std::set<EntityId> Distinct{*FuncId, *RetId, *ParmId};
  EXPECT_EQ(Distinct.size(), 3u);
}

TEST_F(EntitySourceLocationExtractorTest, SystemHeaderDeclSkipped) {
  PathString SysDir = makePath("sysinc");
  std::error_code EC = llvm::sys::fs::create_directory(SysDir);
  ASSERT_FALSE(EC) << EC.message();
  PathString SysHeader = SysDir;
  llvm::sys::path::append(SysHeader, "sysheader.h");
  writeFile(SysHeader, "static int sys_x;\n");

  PathString SrcPath = makePath("sysheader-test.cpp");
  std::string Code = "#include <sysheader.h>\n"
                     "int local_y;\n";
  writeFile(SrcPath, Code);
  runExtractor(SrcPath, Code, {("-isystem" + SysDir.str()).str()});

  const auto *DSys = findDeclByName("sys_x");
  ASSERT_NE(DSys, nullptr);
  EXPECT_TRUE(recordsForDecl(DSys).empty())
      << "system-header decl should not have a recorded source location";

  const auto *DLocal = findDeclByName("local_y");
  ASSERT_NE(DLocal, nullptr);
  EXPECT_FALSE(recordsForDecl(DLocal).empty());
}

} // namespace

} // namespace clang::ssaf
