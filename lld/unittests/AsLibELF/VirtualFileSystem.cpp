//===- VirtualFileSystem.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualFileSystem.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/AssignGUID.h"
#include "gmock/gmock.h"
#include <cstdlib>

using namespace llvm;
using testing::HasSubstr;

LLD_HAS_DRIVER(elf)

namespace {
bool hasAMDGPUTarget() {
#define LLVM_TARGET(TargetName)                                                \
  if (StringRef(#TargetName) == "AMDGPU")                                      \
    return true;
#include "llvm/Config/Targets.def"
  return false;
}

class VirtualFileSystemTest : public testing::Test {
protected:
  SmallString<256> tempDir;
  SmallString<256> inputDir;
  SmallString<256> outputPath;
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> fs =
      makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  std::unique_ptr<MemoryBuffer> object;
  std::string diagnostics;

  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("lld-vfs", tempDir));
    inputDir = tempDir;
    sys::path::append(inputDir, "virtual");
    ASSERT_FALSE(fs->setCurrentWorkingDirectory(inputDir));
    outputPath = tempDir;
    sys::path::append(outputPath, "out.so");

    const char *sourceDir = std::getenv("LLD_SRC_DIR");
    ASSERT_NE(sourceDir, nullptr);
    SmallString<256> path(sourceDir);
    sys::path::append(path, "unittests", "AsLibELF", "Inputs", "kernel1.o");
    auto buffer = MemoryBuffer::getFile(path);
    ASSERT_TRUE(buffer) << buffer.getError().message();
    object = std::move(*buffer);
  }

  void TearDown() override {
    if (!tempDir.empty())
      EXPECT_FALSE(sys::fs::remove_directories(tempDir));
  }

  void addFile(StringRef name, StringRef contents) {
    // Deliberately leave the buffer identifier empty: paths must be resolved
    // using the virtual filename, independently of the backing buffer's name.
    ASSERT_TRUE(fs->addFile(name, 0, MemoryBuffer::getMemBufferCopy(contents)));
  }

  void addObject(StringRef name = "input.o") {
    addFile(name, object->getBuffer());
    SmallString<256> path(name);
    ASSERT_FALSE(fs->makeAbsolute(path));
    ASSERT_FALSE(sys::fs::exists(path));
  }

  lld::Result link(ArrayRef<const char *> inputs, vfs::FileSystem *inputFS) {
    SmallVector<const char *> args = {"ld.lld", "--threads=2", "-o",
                                      outputPath.c_str()};
    if (!llvm::is_contained(inputs, StringRef("-r")))
      args.push_back("-shared");
    args.append(inputs.begin(), inputs.end());
    diagnostics.clear();
    raw_string_ostream err(diagnostics);
    return lld::lldMain(args, nulls(), err, {{lld::Gnu, &lld::elf::link}},
                        inputFS);
  }

  void expectSuccess(ArrayRef<const char *> inputs,
                     vfs::FileSystem *inputFS = nullptr,
                     StringRef symbol = "main_kernel") {
    lld::Result result = link(inputs, inputFS ? inputFS : fs.get());
    ASSERT_EQ(result.retCode, 0) << diagnostics;
    ASSERT_TRUE(result.canRunAgain);
    auto output = object::ObjectFile::createObjectFile(outputPath);
    ASSERT_TRUE(bool(output)) << toString(output.takeError());
    EXPECT_EQ(output->getBinary()->getArch(), Triple::amdgpu);
    bool found = false;
    for (const object::SymbolRef &sym : output->getBinary()->symbols()) {
      auto name = sym.getName();
      ASSERT_TRUE(bool(name)) << toString(name.takeError());
      found |= *name == symbol;
    }
    EXPECT_TRUE(found) << "missing symbol " << symbol;
  }

  void addArchive(bool thin) {
    SmallVector<NewArchiveMember, 1> members;
    members.emplace_back(object->getMemBufferRef());
    members.back().MemberName = "member.o";
    auto archive = writeArchiveToBuffer(
        members, SymtabWritingMode::NormalSymtab, object::Archive::K_GNU,
        /*Deterministic=*/true, thin);
    ASSERT_TRUE(bool(archive)) << toString(archive.takeError());
    ASSERT_TRUE(fs->addFile("lib/libtest.a", 0, std::move(*archive)));
    if (thin)
      addObject("lib/member.o");
  }

  void addBitcode(bool thin) {
    LLVMContext context;
    SMDiagnostic error;
    auto module = parseAssemblyString(R"(
      target triple = "amdgpu9.00-amd-amdhsa"
      target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
      define void @f() {
        ret void
      }
    )",
                                      error, context);
    ASSERT_TRUE(module);
    SmallString<0> bitcode;
    raw_svector_ostream stream(bitcode);
    if (thin) {
      AssignGUIDPass::runOnModule(*module);
      ProfileSummaryInfo psi(*module);
      ModuleSummaryIndex index =
          buildModuleSummaryIndex(*module, nullptr, &psi);
      WriteBitcodeToFile(*module, stream, /*ShouldPreserveUseListOrder=*/false,
                         &index, /*GenerateHash=*/true);
    } else {
      WriteBitcodeToFile(*module, stream);
    }
    addFile("input.bc", bitcode);
  }
};

TEST_F(VirtualFileSystemTest, Object) {
  addObject();
  // Also exercise ELF selection when ld.lld defaults to the MinGW driver.
  expectSuccess({"-m", "elf64_amdgpu", "input.o"});
}

TEST_F(VirtualFileSystemTest, RelocatableOutput) {
  addObject();
  expectSuccess({"-m", "elf64_amdgpu", "-r", "input.o"});
}

TEST_F(VirtualFileSystemTest, ArchiveSearch) {
  addArchive(/*thin=*/false);
  expectSuccess({"-m", "elf64_amdgpu", "--whole-archive", "-Llib", "-ltest"});
  expectSuccess(
      {"-m", "elf64_amdgpu", "--whole-archive", "-Llib", "-l:libtest.a"});
  expectSuccess({"-m", "elf64_amdgpu", "-u", "main_kernel", "-Llib", "-ltest"});
}

TEST_F(VirtualFileSystemTest, ThinArchive) {
  addArchive(/*thin=*/true);
  expectSuccess({"-m", "elf64_amdgpu", "--whole-archive", "lib/libtest.a"});
  expectSuccess({"-m", "elf64_amdgpu", "-u", "main_kernel", "-Llib", "-ltest"});
}

TEST_F(VirtualFileSystemTest, BitcodeAndSampleProfile) {
  if (!hasAMDGPUTarget())
    GTEST_SKIP() << "AMDGPU backend not built";
  addBitcode(/*thin=*/false);
  addFile("profile.prof", "f:0:0\n");
  expectSuccess({"-m", "elf64_amdgpu", "--plugin-opt=mcpu=gfx900", "input.bc",
                 "--lto-sample-profile=profile.prof"},
                nullptr, "f");
}

TEST_F(VirtualFileSystemTest, ThinLTOCacheUsesVirtualProfileContents) {
  if (!hasAMDGPUTarget())
    GTEST_SKIP() << "AMDGPU backend not built";
  addBitcode(/*thin=*/true);
  addFile("profile.prof", "f:0:0\n");
  SmallString<256> cacheDir(tempDir);
  sys::path::append(cacheDir, "cache");
  std::string cacheArg = ("--thinlto-cache-dir=" + cacheDir).str();
  auto linkWithCache = [&] {
    expectSuccess({"-m", "elf64_amdgpu", "--plugin-opt=mcpu=gfx900", "input.bc",
                   "--lto-sample-profile=profile.prof", cacheArg.c_str()},
                  nullptr, "f");
  };
  auto countEntries = [&] {
    unsigned count = 0;
    std::error_code ec;
    for (sys::fs::directory_iterator it(cacheDir, ec), end; it != end && !ec;
         it.increment(ec))
      if (sys::path::filename(it->path()).starts_with("llvmcache-"))
        ++count;
    EXPECT_FALSE(ec);
    return count;
  };
  linkWithCache();
  unsigned entries = countEntries();
  ASSERT_GT(entries, 0u);
  linkWithCache();
  EXPECT_EQ(countEntries(), entries);

  // Keep the paths and bitcode identical, but change the virtual profile.
  fs = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  ASSERT_FALSE(fs->setCurrentWorkingDirectory(inputDir));
  addBitcode(/*thin=*/true);
  addFile("profile.prof", "f:100:0\n 1: 100\n");
  linkWithCache();
  EXPECT_GT(countEntries(), entries);
}

TEST_F(VirtualFileSystemTest, ScriptSearchAndInclude) {
  addObject("scripts/member.o");
  addFile("scripts/main.ld", "INCLUDE fragment.ld\n");
  addFile("scripts/fragment.ld", "GROUP(member.o)\n");
  expectSuccess({"-m", "elf64_amdgpu", "-Lscripts", "-Tmain.ld"});
}

TEST_F(VirtualFileSystemTest, ScriptUnderSysroot) {
  addObject("sysroot/lib/member.o");
  addFile("sysroot/main.ld", "INPUT(/lib/member.o)\n");
  SmallString<256> sysroot(inputDir);
  sys::path::append(sysroot, "sysroot");
  expectSuccess({"-m", "elf64_amdgpu", "--sysroot", sysroot.c_str(),
                 "-Tsysroot/main.ld"});
}

TEST_F(VirtualFileSystemTest, VersionScript) {
  addObject();
  addFile("scripts/versions", "VERSION_1 { global: *; };\n");
  expectSuccess({"-m", "elf64_amdgpu", "input.o", "-Lscripts",
                 "--version-script=versions"});
}

TEST_F(VirtualFileSystemTest, NestedResponseFiles) {
  addObject("input with spaces.o");
  addFile("outer.rsp", "@inner.rsp\n");
  addFile("inner.rsp", "-m elf64_amdgpu \"input with spaces.o\"\n");
  // Both flavor selection and ELF option parsing must expand the response
  // files through the supplied filesystem.
  expectSuccess({"@outer.rsp"});
}

TEST_F(VirtualFileSystemTest, MissingResponseFile) {
  // An explicit emulation skips response-file expansion in flavor selection.
  // The ELF parser must report failures through the supplied error stream.
  lld::Result result = link({"-m", "elf64_amdgpu", "@missing.rsp"}, fs.get());
  EXPECT_NE(result.retCode, 0);
  EXPECT_TRUE(result.canRunAgain);
  EXPECT_THAT(diagnostics, HasSubstr("missing.rsp"));

  result = link({"@missing.rsp"}, fs.get());
  EXPECT_NE(result.retCode, 0);
  EXPECT_TRUE(result.canRunAgain);
  EXPECT_THAT(diagnostics, HasSubstr("missing.rsp"));
}

TEST_F(VirtualFileSystemTest, ReentryUsesNewFilesystem) {
  addObject();
  expectSuccess({"-m", "elf64_amdgpu", "input.o"});

  auto empty = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  ASSERT_FALSE(empty->setCurrentWorkingDirectory(inputDir));
  lld::Result result = link({"-m", "elf64_amdgpu", "input.o"}, empty.get());
  EXPECT_NE(result.retCode, 0);
  EXPECT_TRUE(result.canRunAgain);
  EXPECT_THAT(diagnostics, HasSubstr("cannot open input.o"));

  expectSuccess({"-m", "elf64_amdgpu", "input.o"});
}

TEST_F(VirtualFileSystemTest, OverlayPrecedenceAndFallback) {
  SmallString<256> diskPath(tempDir);
  sys::path::append(diskPath, "disk.o");
  {
    std::error_code ec;
    raw_fd_ostream out(diskPath, ec);
    ASSERT_FALSE(ec);
    out << object->getBuffer();
  }
  auto overlay =
      makeIntrusiveRefCnt<vfs::OverlayFileSystem>(vfs::getRealFileSystem());
  overlay->pushOverlay(fs);
  expectSuccess({"-m", "elf64_amdgpu", diskPath.c_str()}, overlay.get());

  addFile(diskPath, "this is not an object file");
  lld::Result result =
      link({"-m", "elf64_amdgpu", diskPath.c_str()}, overlay.get());
  EXPECT_NE(result.retCode, 0);
  EXPECT_TRUE(result.canRunAgain);
  EXPECT_THAT(diagnostics, HasSubstr("unknown directive"));
}

TEST_F(VirtualFileSystemTest, ReproducerUsesVirtualWorkingDirectory) {
  addObject();
  SmallString<256> tarPath(tempDir);
  sys::path::append(tarPath, "repro.tar");
  expectSuccess(
      {"-m", "elf64_amdgpu", "input.o", "--reproduce", tarPath.c_str()});
  auto tar = MemoryBuffer::getFile(tarPath);
  ASSERT_TRUE(tar) << tar.getError().message();
  // Both response.txt and the archived member must include the virtual
  // working directory, which does not exist on the host filesystem.
  StringRef contents = (*tar)->getBuffer();
  size_t first = contents.find("virtual/input.o");
  ASSERT_NE(first, StringRef::npos);
  EXPECT_NE(contents.find("virtual/input.o", first + 1), StringRef::npos);
}
} // namespace
