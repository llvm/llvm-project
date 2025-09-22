#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

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
    {OffloadArch::UNUSED, "", ""},
    SM(20), {OffloadArch::SM_21, "sm_21", "compute_20"}, // Fermi
    SM(30), {OffloadArch::SM_32_, "sm_32", "compute_32"}, SM(35), SM(37),  // Kepler
    SM(50), SM(52), SM(53),          // Maxwell
    SM(60), SM(61), SM(62),          // Pascal
    SM(70), SM(72),                  // Volta
    SM(75),                          // Turing
    SM(80), SM(86),                  // Ampere
    SM(87),                          // Jetson/Drive AGX Orin
    SM(89),                          // Ada Lovelace
    SM(90),                          // Hopper
    SM(90a),                         // Hopper
    SM(100),                         // Blackwell
    SM(100a),                        // Blackwell
    SM(101),                         // Blackwell
    SM(101a),                        // Blackwell
    SM(103),                         // Blackwell
    SM(103a),                        // Blackwell
    SM(120),                         // Blackwell
    SM(120a),                        // Blackwell
    SM(121),                         // Blackwell
    SM(121a),                        // Blackwell
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
    {OffloadArch::GFX12_GENERIC, "gfx12-generic", "compute_amdgcn"},
    GFX(1200), // gfx1200
    GFX(1201), // gfx1201
    GFX(1250), // gfx1250
    GFX(1251), // gfx1251
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
    return OffloadArch::UNKNOWN;
  return Result->Arch;
}

} // namespace clang
