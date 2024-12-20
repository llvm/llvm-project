//=== ParseHLSLRootSignatureTest.cpp - Parse Root Signature tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/ParseHLSLRootSignature.h"
#include "gtest/gtest.h"

using namespace llvm::hlsl::root_signature;

namespace {

TEST(ParseHLSLRootSignature, EmptyRootFlags) {
  llvm::StringRef RootFlagString = " RootFlags()";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(RootFlagString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);
  ASSERT_EQ(RootFlags::None, RootElements[0].Flags);
}

TEST(ParseHLSLRootSignature, RootFlagsNone) {
  llvm::StringRef RootFlagString = " RootFlags(0)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(RootFlagString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);
  ASSERT_EQ(RootFlags::None, RootElements[0].Flags);
}

TEST(ParseHLSLRootSignature, ValidRootFlags) {
  // Test that the flags are all captured and that they are case insensitive
  llvm::StringRef RootFlagString = " RootFlags( "
                                   " ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT"
                                   "| deny_vertex_shader_root_access"
                                   "| DENY_HULL_SHADER_ROOT_ACCESS"
                                   "| deny_domain_shader_root_access"
                                   "| DENY_GEOMETRY_SHADER_ROOT_ACCESS"
                                   "| deny_pixel_shader_root_access"
                                   "| ALLOW_STREAM_OUTPUT"
                                   "| LOCAL_ROOT_SIGNATURE"
                                   "| deny_amplification_shader_root_access"
                                   "| DENY_MESH_SHADER_ROOT_ACCESS"
                                   "| cbv_srv_uav_heap_directly_indexed"
                                   "| SAMPLER_HEAP_DIRECTLY_INDEXED"
                                   "| AllowLowTierReservedHwCbLimit )";

  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(RootFlagString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);
  ASSERT_EQ(RootFlags::ValidFlags, RootElements[0].Flags);
}

} // anonymous namespace
