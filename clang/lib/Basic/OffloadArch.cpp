//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {

namespace {
struct OffloadArchToStringMap {
  OffloadArch Arch;
  const char *ArchName;
  const char *VirtualArchName;
};
} // namespace

#define SM(sm) {OffloadArch::SM_##sm, "sm_" #sm, "compute_" #sm}
#define GFX(gpu) {OffloadArch::GFX##gpu, "gfx" #gpu, "compute_amdgcn"}
static const OffloadArchToStringMap ArchNames[] = {
    // clang-format off
    {OffloadArch::Unused, "", ""},
    SM(20), {OffloadArch::SM_21, "sm_21", "compute_20"}, // Fermi
    SM(30), {OffloadArch::SM_32_, "sm_32", "compute_32"}, SM(35), SM(37),  // Kepler
    SM(50), SM(52), SM(53),          // Maxwell
    SM(60), SM(61), SM(62),          // Pascal
    SM(70), SM(72),                  // Volta
    SM(75),                          // Turing
    SM(80), SM(86),                  // Ampere
    SM(87),                          // Jetson/Drive AGX Orin
    SM(88),                          // Ampere
    SM(89),                          // Ada Lovelace
    SM(90),                          // Hopper
    SM(90a),                         // Hopper
    SM(100),                         // Blackwell
    SM(100a),                        // Blackwell
    SM(100f),                        // Blackwell
    SM(101),                         // Blackwell
    SM(101a),                        // Blackwell
    SM(101f),                        // Blackwell
    SM(103),                         // Blackwell
    SM(103a),                        // Blackwell
    SM(103f),                        // Blackwell
    SM(110),                         // Blackwell
    SM(110a),                        // Blackwell
    SM(110f),                        // Blackwell
    SM(120),                         // Blackwell
    SM(120a),                        // Blackwell
    SM(120f),                        // Blackwell
    SM(121),                         // Blackwell
    SM(121a),                        // Blackwell
    SM(121f),                        // Blackwell
    GFX(600),  // gfx600
    GFX(601),  // gfx601
    GFX(602),  // gfx602
    GFX(700),  // gfx700
    GFX(701),  // gfx701
    GFX(702),  // gfx702
    GFX(703),  // gfx703
    GFX(704),  // gfx704
    GFX(705),  // gfx705
    GFX(801),  // gfx801
    GFX(802),  // gfx802
    GFX(803),  // gfx803
    GFX(805),  // gfx805
    GFX(810),  // gfx810
    {OffloadArch::GFX9_GENERIC, "gfx9-generic", "compute_amdgcn"},
    GFX(900),  // gfx900
    GFX(902),  // gfx902
    GFX(904),  // gfx903
    GFX(906),  // gfx906
    GFX(908),  // gfx908
    GFX(909),  // gfx909
    GFX(90a),  // gfx90a
    GFX(90c),  // gfx90c
    {OffloadArch::GFX9_4_GENERIC, "gfx9-4-generic", "compute_amdgcn"},
    GFX(942),  // gfx942
    GFX(950),  // gfx950
    {OffloadArch::GFX10_1_GENERIC, "gfx10-1-generic", "compute_amdgcn"},
    GFX(1010), // gfx1010
    GFX(1011), // gfx1011
    GFX(1012), // gfx1012
    GFX(1013), // gfx1013
    {OffloadArch::GFX10_3_GENERIC, "gfx10-3-generic", "compute_amdgcn"},
    GFX(1030), // gfx1030
    GFX(1031), // gfx1031
    GFX(1032), // gfx1032
    GFX(1033), // gfx1033
    GFX(1034), // gfx1034
    GFX(1035), // gfx1035
    GFX(1036), // gfx1036
    {OffloadArch::GFX11_GENERIC, "gfx11-generic", "compute_amdgcn"},
    GFX(1100), // gfx1100
    GFX(1101), // gfx1101
    GFX(1102), // gfx1102
    GFX(1103), // gfx1103
    GFX(1150), // gfx1150
    GFX(1151), // gfx1151
    GFX(1152), // gfx1152
    GFX(1153), // gfx1153
    GFX(1154), // gfx1154
    {OffloadArch::GFX11_7_GENERIC, "gfx11-7-generic", "compute_amdgcn"},
    GFX(1170), // gfx1170
    GFX(1171), // gfx1171
    GFX(1172), // gfx1172
    {OffloadArch::GFX12_GENERIC, "gfx12-generic", "compute_amdgcn"},
    GFX(1200), // gfx1200
    GFX(1201), // gfx1201
    {OffloadArch::GFX12_5_GENERIC, "gfx12-5-generic", "compute_amdgcn"},
    GFX(1250), // gfx1250
    GFX(1251), // gfx1251
    {OffloadArch::GFX13_GENERIC, "gfx13-generic", "compute_amdgcn"},
    GFX(1310), // gfx1310
    {OffloadArch::AMDGCNSPIRV, "amdgcnspirv", "compute_amdgcn"},
    // Intel CPUs
    {OffloadArch::GRANITERAPIDS, "graniterapids", ""},
    // Intel GPUS
    {OffloadArch::BMG_G21, "bmg_g21", ""},
    {OffloadArch::Generic, "generic", ""},
    // clang-format on
};
#undef SM
#undef GFX

const char *OffloadArchToString(OffloadArch A) {
  auto Result =
      llvm::find_if(ArchNames, [A](const OffloadArchToStringMap &Map) {
        return A == Map.Arch;
      });
  if (Result == std::end(ArchNames))
    return "unknown";
  return Result->ArchName;
}

const char *OffloadArchToVirtualArchString(OffloadArch A) {
  auto Result =
      llvm::find_if(ArchNames, [A](const OffloadArchToStringMap &Map) {
        return A == Map.Arch;
      });
  if (Result == std::end(ArchNames))
    return "unknown";
  return Result->VirtualArchName;
}

OffloadArch StringToOffloadArch(llvm::StringRef S) {
  auto Result =
      llvm::find_if(ArchNames, [S](const OffloadArchToStringMap &Map) {
        return S == Map.ArchName;
      });
  if (Result == std::end(ArchNames))
    return OffloadArch::Unknown;
  return Result->Arch;
}

OffloadArch getSubArchOffloadArch(llvm::Triple::SubArchType SubArch) {
  static const OffloadArch AMDGPUSubArchs[llvm::Triple::LastAMDGPUSubArch -
                                          llvm::Triple::FirstAMDGPUSubArch +
                                          1] = {
      OffloadArch::Unknown,         OffloadArch::GFX600,  OffloadArch::GFX601,
      OffloadArch::GFX602,

      OffloadArch::Unknown,         OffloadArch::GFX700,  OffloadArch::GFX701,
      OffloadArch::GFX702,          OffloadArch::GFX703,  OffloadArch::GFX704,
      OffloadArch::GFX705,

      OffloadArch::Unknown,         OffloadArch::GFX801,  OffloadArch::GFX802,
      OffloadArch::GFX803,          OffloadArch::GFX805,

      OffloadArch::GFX810,

      OffloadArch::GFX9_GENERIC,    OffloadArch::GFX900,  OffloadArch::GFX902,
      OffloadArch::GFX904,          OffloadArch::GFX906,  OffloadArch::GFX909,
      OffloadArch::GFX90c,

      OffloadArch::GFX908,

      OffloadArch::GFX90a,

      OffloadArch::GFX9_4_GENERIC,  OffloadArch::GFX942,  OffloadArch::GFX950,

      OffloadArch::GFX10_1_GENERIC, OffloadArch::GFX1010, OffloadArch::GFX1011,
      OffloadArch::GFX1012,         OffloadArch::GFX1013,

      OffloadArch::GFX10_3_GENERIC, OffloadArch::GFX1030, OffloadArch::GFX1031,
      OffloadArch::GFX1032,         OffloadArch::GFX1033, OffloadArch::GFX1034,
      OffloadArch::GFX1035,         OffloadArch::GFX1036,

      OffloadArch::GFX11_GENERIC,   OffloadArch::GFX1100, OffloadArch::GFX1101,
      OffloadArch::GFX1102,         OffloadArch::GFX1103, OffloadArch::GFX1150,
      OffloadArch::GFX1151,         OffloadArch::GFX1152, OffloadArch::GFX1153,
      OffloadArch::GFX1154,

      OffloadArch::GFX11_7_GENERIC, OffloadArch::GFX1170, OffloadArch::GFX1171,
      OffloadArch::GFX1172,

      OffloadArch::GFX12_GENERIC,   OffloadArch::GFX1200, OffloadArch::GFX1201,

      OffloadArch::GFX12_5_GENERIC, OffloadArch::GFX1250, OffloadArch::GFX1251,

      OffloadArch::GFX13_GENERIC,   OffloadArch::GFX1310};

  if (SubArch < llvm::Triple::FirstAMDGPUSubArch ||
      SubArch > llvm::Triple::LastAMDGPUSubArch)
    return OffloadArch::Unknown;

  return AMDGPUSubArchs[SubArch - llvm::Triple::FirstAMDGPUSubArch];
}

llvm::Triple::SubArchType getOffloadArchSubArch(OffloadArch ID) {
  switch (ID) {
  case OffloadArch::Unused:
  case OffloadArch::Unknown:
  case OffloadArch::SM_20:
  case OffloadArch::SM_21:
  case OffloadArch::SM_30:
  case OffloadArch::SM_32_:
  case OffloadArch::SM_35:
  case OffloadArch::SM_37:
  case OffloadArch::SM_50:
  case OffloadArch::SM_52:
  case OffloadArch::SM_53:
  case OffloadArch::SM_60:
  case OffloadArch::SM_61:
  case OffloadArch::SM_62:
  case OffloadArch::SM_70:
  case OffloadArch::SM_72:
  case OffloadArch::SM_75:
  case OffloadArch::SM_80:
  case OffloadArch::SM_86:
  case OffloadArch::SM_87:
  case OffloadArch::SM_88:
  case OffloadArch::SM_89:
  case OffloadArch::SM_90:
  case OffloadArch::SM_90a:
  case OffloadArch::SM_100:
  case OffloadArch::SM_100a:
  case OffloadArch::SM_100f:
  case OffloadArch::SM_101:
  case OffloadArch::SM_101a:
  case OffloadArch::SM_101f:
  case OffloadArch::SM_103:
  case OffloadArch::SM_103a:
  case OffloadArch::SM_103f:
  case OffloadArch::SM_110:
  case OffloadArch::SM_110a:
  case OffloadArch::SM_110f:
  case OffloadArch::SM_120:
  case OffloadArch::SM_120a:
  case OffloadArch::SM_120f:
  case OffloadArch::SM_121:
  case OffloadArch::SM_121a:
  case OffloadArch::SM_121f:
  case OffloadArch::AMDGCNSPIRV:
  case OffloadArch::Generic:
  case OffloadArch::GRANITERAPIDS:
  case OffloadArch::BMG_G21:
    return llvm::Triple::NoSubArch;
  case OffloadArch::GFX600:
    return llvm::Triple::AMDGPUSubArch600;
  case OffloadArch::GFX601:
    return llvm::Triple::AMDGPUSubArch601;
  case OffloadArch::GFX602:
    return llvm::Triple::AMDGPUSubArch602;
  case OffloadArch::GFX700:
    return llvm::Triple::AMDGPUSubArch700;
  case OffloadArch::GFX701:
    return llvm::Triple::AMDGPUSubArch701;
  case OffloadArch::GFX702:
    return llvm::Triple::AMDGPUSubArch702;
  case OffloadArch::GFX703:
    return llvm::Triple::AMDGPUSubArch703;
  case OffloadArch::GFX704:
    return llvm::Triple::AMDGPUSubArch704;
  case OffloadArch::GFX705:
    return llvm::Triple::AMDGPUSubArch705;
  case OffloadArch::GFX801:
    return llvm::Triple::AMDGPUSubArch801;
  case OffloadArch::GFX802:
    return llvm::Triple::AMDGPUSubArch802;
  case OffloadArch::GFX803:
    return llvm::Triple::AMDGPUSubArch803;
  case OffloadArch::GFX805:
    return llvm::Triple::AMDGPUSubArch805;
  case OffloadArch::GFX810:
    return llvm::Triple::AMDGPUSubArch810;
  case OffloadArch::GFX9_GENERIC:
    return llvm::Triple::AMDGPUSubArch9;
  case OffloadArch::GFX900:
    return llvm::Triple::AMDGPUSubArch900;
  case OffloadArch::GFX902:
    return llvm::Triple::AMDGPUSubArch902;
  case OffloadArch::GFX904:
    return llvm::Triple::AMDGPUSubArch904;
  case OffloadArch::GFX906:
    return llvm::Triple::AMDGPUSubArch906;
  case OffloadArch::GFX908:
    return llvm::Triple::AMDGPUSubArch908;
  case OffloadArch::GFX909:
    return llvm::Triple::AMDGPUSubArch909;
  case OffloadArch::GFX90a:
    return llvm::Triple::AMDGPUSubArch90A;
  case OffloadArch::GFX90c:
    return llvm::Triple::AMDGPUSubArch90C;
  case OffloadArch::GFX9_4_GENERIC:
    return llvm::Triple::AMDGPUSubArch9_4;
  case OffloadArch::GFX942:
    return llvm::Triple::AMDGPUSubArch942;
  case OffloadArch::GFX950:
    return llvm::Triple::AMDGPUSubArch950;
  case OffloadArch::GFX10_1_GENERIC:
    return llvm::Triple::AMDGPUSubArch10_1;
  case OffloadArch::GFX1010:
    return llvm::Triple::AMDGPUSubArch1010;
  case OffloadArch::GFX1011:
    return llvm::Triple::AMDGPUSubArch1011;
  case OffloadArch::GFX1012:
    return llvm::Triple::AMDGPUSubArch1012;
  case OffloadArch::GFX1013:
    return llvm::Triple::AMDGPUSubArch1013;
  case OffloadArch::GFX10_3_GENERIC:
    return llvm::Triple::AMDGPUSubArch10_3;
  case OffloadArch::GFX1030:
    return llvm::Triple::AMDGPUSubArch1030;
  case OffloadArch::GFX1031:
    return llvm::Triple::AMDGPUSubArch1031;
  case OffloadArch::GFX1032:
    return llvm::Triple::AMDGPUSubArch1032;
  case OffloadArch::GFX1033:
    return llvm::Triple::AMDGPUSubArch1033;
  case OffloadArch::GFX1034:
    return llvm::Triple::AMDGPUSubArch1034;
  case OffloadArch::GFX1035:
    return llvm::Triple::AMDGPUSubArch1035;
  case OffloadArch::GFX1036:
    return llvm::Triple::AMDGPUSubArch1036;
  case OffloadArch::GFX11_GENERIC:
    return llvm::Triple::AMDGPUSubArch11;
  case OffloadArch::GFX1100:
    return llvm::Triple::AMDGPUSubArch1100;
  case OffloadArch::GFX1101:
    return llvm::Triple::AMDGPUSubArch1101;
  case OffloadArch::GFX1102:
    return llvm::Triple::AMDGPUSubArch1102;
  case OffloadArch::GFX1103:
    return llvm::Triple::AMDGPUSubArch1103;
  case OffloadArch::GFX1150:
    return llvm::Triple::AMDGPUSubArch1150;
  case OffloadArch::GFX1151:
    return llvm::Triple::AMDGPUSubArch1151;
  case OffloadArch::GFX1152:
    return llvm::Triple::AMDGPUSubArch1152;
  case OffloadArch::GFX1153:
    return llvm::Triple::AMDGPUSubArch1153;
  case OffloadArch::GFX1154:
    return llvm::Triple::AMDGPUSubArch1154;
  case OffloadArch::GFX1170:
    return llvm::Triple::AMDGPUSubArch1170;
  case OffloadArch::GFX11_7_GENERIC:
    return llvm::Triple::AMDGPUSubArch11_7;
  case OffloadArch::GFX1171:
    return llvm::Triple::AMDGPUSubArch1171;
  case OffloadArch::GFX1172:
    return llvm::Triple::AMDGPUSubArch1172;
  case OffloadArch::GFX12_GENERIC:
    return llvm::Triple::AMDGPUSubArch12;
  case OffloadArch::GFX1200:
    return llvm::Triple::AMDGPUSubArch1200;
  case OffloadArch::GFX1201:
    return llvm::Triple::AMDGPUSubArch1201;
  case OffloadArch::GFX1250:
    return llvm::Triple::AMDGPUSubArch1250;
  case OffloadArch::GFX1251:
    return llvm::Triple::AMDGPUSubArch1251;
  case OffloadArch::GFX12_5_GENERIC:
    return llvm::Triple::AMDGPUSubArch12_5;
  case OffloadArch::GFX13_GENERIC:
    return llvm::Triple::AMDGPUSubArch13;
  case OffloadArch::GFX1310:
    return llvm::Triple::AMDGPUSubArch1310;
  }

  llvm_unreachable("covered switch");
}

llvm::Triple OffloadArchToTriple(const llvm::Triple &DefaultToolchainTriple,
                                 OffloadArch ID) {
  if (ID == OffloadArch::AMDGCNSPIRV)
    return llvm::Triple(llvm::Triple::spirv64, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  if (IsNVIDIAOffloadArch(ID)) {
    llvm::Triple::ArchType Arch = DefaultToolchainTriple.isArch64Bit()
                                      ? llvm::Triple::nvptx64
                                      : llvm::Triple::nvptx;
    return llvm::Triple(Arch, llvm::Triple::NoSubArch, llvm::Triple::NVIDIA,
                        llvm::Triple::CUDA);
  }

  if (IsAMDOffloadArch(ID)) {
    return llvm::Triple(llvm::Triple::amdgpu, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);
  }

  return {};
}

} // namespace clang
