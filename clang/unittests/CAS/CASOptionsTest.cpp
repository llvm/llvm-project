//===- CASOptionsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CAS/CASOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::cas;

namespace {

TEST(CASOptionsTest, getKind) {
  CASOptions Opts;
  EXPECT_EQ(CASOptions::InMemoryCAS, Opts.getKind());

  if (!llvm::cas::isOnDiskCASEnabled())
    return;
  Opts.CASPath = "auto";
  unittest::TempDir Dir("cas-options", /*Unique=*/true);
  EXPECT_EQ(CASOptions::OnDiskCAS, Opts.getKind());

  Opts.CASPath = Dir.path("cas").str().str();
  EXPECT_EQ(CASOptions::OnDiskCAS, Opts.getKind());
}

TEST(CASOptionsTest, getOrCreateDatabases) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());

  // Create an in-memory CAS.
  CASOptions Opts;
  auto [InMemoryCAS, InMemoryAC] = Opts.getOrCreateDatabases(Diags);
  ASSERT_TRUE(InMemoryCAS);
  ASSERT_TRUE(InMemoryAC);
  EXPECT_EQ(InMemoryCAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(InMemoryAC, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::InMemoryCAS, Opts.getKind());

  if (!llvm::cas::isOnDiskCASEnabled())
    return;

  // Create an on-disk CAS.
  unittest::TempDir Dir("cas-options", /*Unique=*/true);
  Opts.CASPath = Dir.path("cas").str().str();
  auto [OnDiskCAS, OnDiskAC] = Opts.getOrCreateDatabases(Diags);
  EXPECT_NE(InMemoryCAS, OnDiskCAS);
  EXPECT_NE(InMemoryAC, OnDiskAC);
  EXPECT_EQ(OnDiskCAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(OnDiskAC, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::OnDiskCAS, Opts.getKind());

  // Create an on-disk CAS at an automatic location.
  Opts.CASPath = "auto";
  auto [AutoCAS, AutoAC] = Opts.getOrCreateDatabases(Diags);
  EXPECT_NE(InMemoryCAS, AutoCAS);
  EXPECT_NE(InMemoryAC, AutoAC);
  EXPECT_NE(OnDiskCAS, AutoCAS);
  EXPECT_NE(OnDiskAC, AutoAC);
  EXPECT_EQ(AutoCAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(AutoAC, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::OnDiskCAS, Opts.getKind());

  // Create another in-memory CAS. It won't be the same one.
  Opts.CASPath = "";
  auto [InMemoryCAS2, InMemoryAC2] = Opts.getOrCreateDatabases(Diags);
  EXPECT_NE(InMemoryCAS, InMemoryCAS2);
  EXPECT_NE(InMemoryAC, InMemoryAC2);
  EXPECT_NE(OnDiskCAS, InMemoryCAS2);
  EXPECT_NE(OnDiskAC, InMemoryAC2);
  EXPECT_NE(AutoCAS, InMemoryCAS2);
  EXPECT_NE(AutoAC, InMemoryAC2);
  EXPECT_EQ(InMemoryCAS2, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(InMemoryAC2, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::InMemoryCAS, Opts.getKind());
}

TEST(CASOptionsTest, getOrCreateObjectStoreInvalid) {
  if (!llvm::cas::isOnDiskCASEnabled())
    return;

  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());

  // Create a file, then try to put a CAS there.
  StringRef Contents = "contents";
  unittest::TempDir Dir("cas-options", /*Unique=*/true);
  unittest::TempFile File(Dir.path("cas"), /*Suffix=*/"",
                          /*Contents=*/Contents);

  CASOptions Opts;
  Opts.CASPath = File.path().str();
  EXPECT_EQ(nullptr, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(nullptr, Opts.getOrCreateDatabases(Diags).second);

  auto [EmptyCAS, EmptyAC] =
      Opts.getOrCreateDatabases(Diags, /*CreateEmptyCASOnFailure=*/true);
  EXPECT_EQ(EmptyCAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(EmptyAC, Opts.getOrCreateDatabases(Diags).second);

  // Ensure the file wasn't clobbered.
  std::unique_ptr<MemoryBuffer> MemBuffer;
  ASSERT_THAT_ERROR(
      errorOrToExpected(MemoryBuffer::getFile(File.path())).moveInto(MemBuffer),
      Succeeded());
  ASSERT_EQ(Contents, MemBuffer->getBuffer());
}

TEST(CASOptionsTest, freezeConfig) {
  if (!llvm::cas::isOnDiskCASEnabled())
    return;

  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());

  // Hide the CAS configuration when creating it.
  unittest::TempDir Dir("cas-options", /*Unique=*/true);
  CASOptions Opts;
  Opts.CASPath = Dir.path("cas").str().str();
  Opts.freezeConfig(Diags);
  auto [CAS, AC] = Opts.getOrCreateDatabases(Diags);
  ASSERT_TRUE(CAS);
  ASSERT_TRUE(AC);
  EXPECT_EQ(CASOptions::UnknownCAS, Opts.getKind());

  // Check that the configuration is hidden, but calls to
  // getOrCreateObjectStore() still return the original CAS.
  EXPECT_EQ(CAS->getContext().getHashSchemaIdentifier(), Opts.CASPath);

  // Check that new paths are ignored.
  Opts.CASPath = "";
  EXPECT_EQ(CAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(AC, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::UnknownCAS, Opts.getKind());

  Opts.CASPath = Dir.path("ignored-cas").str().str();
  EXPECT_EQ(CAS, Opts.getOrCreateDatabases(Diags).first);
  EXPECT_EQ(AC, Opts.getOrCreateDatabases(Diags).second);
  EXPECT_EQ(CASOptions::UnknownCAS, Opts.getKind());
}

TEST(CASOptionsTest, equal) {
  CASOptions Opt1, Opt2;
  ASSERT_TRUE(Opt1 == Opt2);

  Opt1.CASPath = "some/path";
  Opt2.CASPath = "some/path";
  ASSERT_TRUE(Opt1 == Opt2);
  Opt2.CASPath = "other/path";
  ASSERT_TRUE(Opt1 != Opt2);

  Opt1.CASPath.clear();
  Opt1.PluginPath = "plugin/path";
  ASSERT_TRUE(Opt1 != Opt2);
  Opt2.CASPath.clear();
  ASSERT_TRUE(Opt1 != Opt2);
  Opt2.PluginPath = "plugin/path2";
  ASSERT_TRUE(Opt1 != Opt2);
  Opt2.PluginPath = "plugin/path";
  ASSERT_TRUE(Opt1 == Opt2);

  Opt1.PluginOptions.emplace_back("key", "value");
  ASSERT_TRUE(Opt1 != Opt2);
  Opt2.PluginOptions.emplace_back("key", "value");
  ASSERT_TRUE(Opt1 == Opt2);
}

} // end namespace
