#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>

namespace clang {

namespace {
struct OffloadArchToStringMap {
  OffloadArch arch;
  const char *arch_name;
  const char *virtual_arch_name;
};
} // namespace

#define SM2(sm, ca) {OffloadArch::SM_##sm, "sm_" #sm, ca}
#define SM(sm) SM2(sm, "compute_" #sm)
#define GFX(gpu) {OffloadArch::GFX##gpu, "gfx" #gpu, "compute_amdgcn"}
static const OffloadArchToStringMap arch_names[] = {
    // clang-format off
    {OffloadArch::UNUSED, "", ""},
    SM2(20, "compute_20"), SM2(21, "compute_20"), // Fermi
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
    SM(120),                         // Blackwell
    SM(120a),                        // Blackwell
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
    {OffloadArch::AMDGCNSPIRV, "amdgcnspirv", "compute_amdgcn"},
    // Note: this is an initial list of Intel GPU and GPU offloading architectures.
    // The list will be expanded later as support for more architectures is added.
    // Intel CPUs
    {OffloadArch::GRANITERAPIDS, "graniterapids", ""},
    // Intel GPUS
    {OffloadArch::BMG_G21, "bmg_g21", ""},
    {OffloadArch::Generic, "generic", ""},
    // clang-format on
};
#undef SM
#undef SM2
#undef GFX
#undef INTEL

const char *OffloadArchToString(OffloadArch A) {
  auto result = std::find_if(
      std::begin(arch_names), std::end(arch_names),
      [A](const OffloadArchToStringMap &map) { return A == map.arch; });
  if (result == std::end(arch_names))
    return "unknown";
  return result->arch_name;
}

const char *OffloadArchToVirtualArchString(OffloadArch A) {
  auto result = std::find_if(
      std::begin(arch_names), std::end(arch_names),
      [A](const OffloadArchToStringMap &map) { return A == map.arch; });
  if (result == std::end(arch_names))
    return "unknown";
  return result->virtual_arch_name;
}

OffloadArch StringToOffloadArch(llvm::StringRef S) {
  auto result = std::find_if(
      std::begin(arch_names), std::end(arch_names),
      [S](const OffloadArchToStringMap &map) { return S == map.arch_name; });
  if (result == std::end(arch_names))
    return OffloadArch::UNKNOWN;
  return result->arch;
}

} // namespace clang
