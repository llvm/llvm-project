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

TEST(ParseHLSLRootSignature, MandatoryRootConstant) {
  llvm::StringRef RootFlagString = "RootConstants(num32BitConstants = 4, b42)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(RootFlagString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::Constants, Parameter.Type);
  ASSERT_EQ(RegisterType::BReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)42, Parameter.Register.Number);
  ASSERT_EQ((uint32_t)4, Parameter.Num32BitConstants);
  ASSERT_EQ((uint32_t)0, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::All, Parameter.Visibility);
}

TEST(ParseHLSLRootSignature, OptionalRootConstant) {
  llvm::StringRef RootFlagString =
      "RootConstants(num32BitConstants = 4, b42, space = 4, visibility = "
      "SHADER_VISIBILITY_DOMAIN)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(RootFlagString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::Constants, Parameter.Type);
  ASSERT_EQ(RegisterType::BReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)42, Parameter.Register.Number);
  ASSERT_EQ((uint32_t)4, Parameter.Num32BitConstants);
  ASSERT_EQ((uint32_t)4, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::Domain, Parameter.Visibility);
}

TEST(ParseHLSLRootSignature, DefaultRootCBV) {
  llvm::StringRef ViewsString = "CBV(b0)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(ViewsString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::CBV, Parameter.Type);
  ASSERT_EQ(RegisterType::BReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)0, Parameter.Register.Number);
  ASSERT_EQ(RootDescriptorFlags::None, Parameter.Flags);
  ASSERT_EQ((uint32_t)0, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::All, Parameter.Visibility);
}

TEST(ParseHLSLRootSignature, SampleRootCBV) {
  llvm::StringRef ViewsString = "CBV(b982374, space = 1, flags = DATA_STATIC)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(ViewsString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::CBV, Parameter.Type);
  ASSERT_EQ(RegisterType::BReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)982374, Parameter.Register.Number);
  ASSERT_EQ(RootDescriptorFlags::DataStatic, Parameter.Flags);
  ASSERT_EQ((uint32_t)1, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::All, Parameter.Visibility);
}

TEST(ParseHLSLRootSignature, SampleRootSRV) {
  llvm::StringRef ViewsString = "SRV(t3, visibility = SHADER_VISIBILITY_MESH, "
                                "flags = Data_Static_While_Set_At_Execute)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(ViewsString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::SRV, Parameter.Type);
  ASSERT_EQ(RegisterType::TReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)3, Parameter.Register.Number);
  ASSERT_EQ(RootDescriptorFlags::DataStaticWhileSetAtExecute, Parameter.Flags);
  ASSERT_EQ((uint32_t)0, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::Mesh, Parameter.Visibility);
}

TEST(ParseHLSLRootSignature, SampleRootUAV) {
  llvm::StringRef ViewsString = "UAV(u0, flags = DATA_VOLATILE)";
  llvm::SmallVector<RootElement> RootElements;
  Parser Parser(ViewsString, &RootElements);
  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ(RootElements.size(), (unsigned long)1);

  RootParameter Parameter = RootElements[0].Parameter;
  ASSERT_EQ(RootType::UAV, Parameter.Type);
  ASSERT_EQ(RegisterType::UReg, Parameter.Register.ViewType);
  ASSERT_EQ((uint32_t)0, Parameter.Register.Number);
  ASSERT_EQ(RootDescriptorFlags::DataVolatile, Parameter.Flags);
  ASSERT_EQ((uint32_t)0, Parameter.Space);
  ASSERT_EQ(ShaderVisibility::All, Parameter.Visibility);
}
} // anonymous namespace
