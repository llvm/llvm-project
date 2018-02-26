//===- unittests/Frontend/CompilerInstanceTest.cpp - CI tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

TEST(CompilerInstance, DefaultVFSOverlayFromInvocation) {
  // Create a temporary VFS overlay yaml file.
  int FD;
  SmallString<256> FileName;
  ASSERT_FALSE(sys::fs::createTemporaryFile("vfs", "yaml", FD, FileName));
  ToolOutputFile File(FileName, FD);

  SmallString<256> CurrentPath;
  sys::fs::current_path(CurrentPath);
  sys::fs::make_absolute(CurrentPath, FileName);

  // Mount the VFS file itself on the path 'virtual.file'. Makes this test
  // a bit shorter than creating a new dummy file just for this purpose.
  const std::string CurrentPathStr = CurrentPath.str();
  const std::string FileNameStr = FileName.str();
  const char *VFSYaml = "{ 'version': 0, 'roots': [\n"
                        "  { 'name': '%s',\n"
                        "    'type': 'directory',\n"
                        "    'contents': [\n"
                        "      { 'name': 'vfs-virtual.file', 'type': 'file',\n"
                        "        'external-contents': '%s'\n"
                        "      }\n"
                        "    ]\n"
                        "  }\n"
                        "]}\n";
  File.os() << format(VFSYaml, CurrentPathStr.c_str(), FileName.c_str());
  File.os().flush();

  // Create a CompilerInvocation that uses this overlay file.
  const std::string VFSArg = "-ivfsoverlay" + FileNameStr;
  const char *Args[] = {"clang", VFSArg.c_str(), "-xc++", "-"};

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());

  std::shared_ptr<CompilerInvocation> CInvok =
      createInvocationFromCommandLine(Args, Diags);

  if (!CInvok)
    FAIL() << "could not create compiler invocation";
  // Create a minimal CompilerInstance which should use the VFS we specified
  // in the CompilerInvocation (as we don't explicitly set our own).
  CompilerInstance Instance;
  Instance.setDiagnostics(Diags.get());
  Instance.setInvocation(CInvok);
  Instance.createFileManager();

  // Check if the virtual file exists which means that our VFS is used by the
  // CompilerInstance.
  ASSERT_TRUE(Instance.getFileManager().getFile("vfs-virtual.file"));
}

} // anonymous namespace
