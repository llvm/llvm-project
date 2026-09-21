//===- unittests/Serialization/UnresolvedSubmoduleTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/Module.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/ModuleFile.h"
#include "clang/Serialization/ModuleManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class UnresolvedSubmoduleTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(
        sys::fs::createUniqueDirectory("unresolved-submodule", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  SmallString<256> TestDir;

  void addFile(StringRef Path, StringRef Contents) {
    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);
    ASSERT_FALSE(sys::fs::create_directories(sys::path::parent_path(AbsPath)));
    std::error_code EC;
    raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }
};

// A header in a module file is associated with the submodule it belongs to when
// its header info is read. If the submodule ID does not resolve, getSubmodule()
// reports the error and returns null, and the association has to be skipped:
// ModuleMap::addHeader() dereferences the Module it is given.
TEST_F(UnresolvedSubmoduleTest, HeaderInfoWithUnresolvableSubmoduleID) {
  addFile("mod/module.modulemap", R"cc(
module A {
  module a1 { header "a1.h" export * }
  module a2 { header "a2.h" export * }
}
)cc");
  addFile("mod/a1.h", "static inline int a1(void) { return 1; }\n");
  addFile("mod/a2.h", "static inline int a2(void) { return 2; }\n");

  SmallString<256> ModuleMap(TestDir);
  sys::path::append(ModuleMap, "mod", "module.modulemap");
  SmallString<256> HeaderPath(TestDir);
  sys::path::append(HeaderPath, "mod", "a1.h");
  SmallString<256> PCMPath(TestDir);
  sys::path::append(PCMPath, "A.pcm");

  {
    CreateInvocationOptions CIOpts;
    CIOpts.VFS = vfs::createPhysicalFileSystem();
    DiagnosticOptions DiagOpts;
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
    CIOpts.Diags = Diags;

    const char *Args[] = {"clang",
                          "-fmodules",
                          "-fno-implicit-modules",
                          "-x",
                          "c",
                          "-Xclang",
                          "-emit-module",
                          "-Xclang",
                          "-fmodule-name=A",
                          ModuleMap.c_str(),
                          "-o",
                          PCMPath.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    ASSERT_TRUE(Invocation);
    Invocation->getFrontendOpts().DisableFree = false;

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags);
    Instance.createVirtualFileSystem(CIOpts.VFS);
    Instance.createFileManager();
    Instance.getFrontendOpts().OutputFile = PCMPath.str().str();

    GenerateModuleFromModuleMapAction Action;
    ASSERT_TRUE(Instance.ExecuteAction(Action));
    ASSERT_FALSE(Diags->hasErrorOccurred());
  }

  CreateInvocationOptions CIOpts;
  CIOpts.VFS = vfs::createPhysicalFileSystem();
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
  CIOpts.Diags = Diags;

  const char *Args[] = {"clang", "-fmodules", "-fno-implicit-modules",
                        "-x",    "c",         "-"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  Invocation->getFrontendOpts().DisableFree = false;

  CompilerInstance Clang(std::move(Invocation));
  Clang.setDiagnostics(Diags);
  Clang.createVirtualFileSystem(CIOpts.VFS);
  Clang.createFileManager();
  Clang.createSourceManager();
  ASSERT_TRUE(Clang.createTarget());
  Clang.createPreprocessor(TU_Complete);
  Clang.createASTContext();
  Clang.createASTReader();

  IntrusiveRefCntPtr<ASTReader> Reader = Clang.getASTReader();
  ASSERT_TRUE(Reader);

  ASSERT_EQ(Reader->ReadAST(ModuleFileName::makeExplicit(PCMPath.str()),
                            serialization::MK_ExplicitModule, SourceLocation(),
                            ASTReader::ARR_None),
            ASTReader::Success);

  serialization::ModuleFile &MF = Reader->getModuleManager().getPrimaryModule();
  ASSERT_GT(MF.LocalNumSubmodules, 0u);

  // Point the module file's local submodule IDs far past the end of
  // SubmodulesLoaded. getSubmodule() diagnoses that and returns null, which is
  // the state a module file with an inconsistent submodule table produces.
  MF.SubmoduleRemap = ContinuousRangeMap<uint32_t, int, 2>();
  {
    ContinuousRangeMap<uint32_t, int, 2>::Builder B(MF.SubmoduleRemap);
    B.insert(std::make_pair(MF.LocalBaseSubmoduleID, 1 << 20));
  }

  auto FE = Clang.getFileManager().getOptionalFileRef(HeaderPath);
  ASSERT_TRUE(FE);

  // Reading the header info walks every loaded module file, reaches this module
  // file's entry for a1.h, and fails to resolve its submodule. This must not
  // dereference the null Module.
  Reader->GetHeaderFileInfo(*FE);
}

} // anonymous namespace
