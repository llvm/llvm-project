if(COMMAND include_guard)
    include_guard(DIRECTORY)
else()
string(MAKE_C_IDENTIFIER "${CMAKE_CURRENT_LIST_FILE}" _PACKAGE_ID)
if(DEFINED ${_GUARD_FILE_${_PACKAGE_ID}})
    return()
endif()
set(${_GUARD_FILE_${_PACKAGE_ID}} On)
endif()



add_library(oclc_abi_version_400 STATIC IMPORTED)
set_target_properties(oclc_abi_version_400 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_400.bc")
add_library(oclc_abi_version_500 STATIC IMPORTED)
set_target_properties(oclc_abi_version_500 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_500.bc")
add_library(oclc_abi_version_600 STATIC IMPORTED)
set_target_properties(oclc_abi_version_600 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_600.bc")
add_library(oclc_correctly_rounded_sqrt_off STATIC IMPORTED)
set_target_properties(oclc_correctly_rounded_sqrt_off PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc")
add_library(oclc_correctly_rounded_sqrt_on STATIC IMPORTED)
set_target_properties(oclc_correctly_rounded_sqrt_on PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc")
add_library(oclc_daz_opt_off STATIC IMPORTED)
set_target_properties(oclc_daz_opt_off PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_daz_opt_off.bc")
add_library(oclc_daz_opt_on STATIC IMPORTED)
set_target_properties(oclc_daz_opt_on PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_daz_opt_on.bc")
add_library(oclc_finite_only_off STATIC IMPORTED)
set_target_properties(oclc_finite_only_off PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_finite_only_off.bc")
add_library(oclc_finite_only_on STATIC IMPORTED)
set_target_properties(oclc_finite_only_on PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_finite_only_on.bc")
add_library(oclc_isa_version_10-1-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_10-1-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_10-1-generic.bc")
add_library(oclc_isa_version_10-3-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_10-3-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_10-3-generic.bc")
add_library(oclc_isa_version_1010 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1010 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1010.bc")
add_library(oclc_isa_version_1011 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1011 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1011.bc")
add_library(oclc_isa_version_1012 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1012 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1012.bc")
add_library(oclc_isa_version_1013 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1013 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1013.bc")
add_library(oclc_isa_version_1030 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1030 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1030.bc")
add_library(oclc_isa_version_1031 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1031 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1031.bc")
add_library(oclc_isa_version_1032 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1032 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1032.bc")
add_library(oclc_isa_version_1033 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1033 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1033.bc")
add_library(oclc_isa_version_1034 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1034 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1034.bc")
add_library(oclc_isa_version_1035 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1035 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1035.bc")
add_library(oclc_isa_version_1036 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1036 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1036.bc")
add_library(oclc_isa_version_11-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_11-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_11-generic.bc")
add_library(oclc_isa_version_1100 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1100 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1100.bc")
add_library(oclc_isa_version_1101 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1101 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1101.bc")
add_library(oclc_isa_version_1102 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1102 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1102.bc")
add_library(oclc_isa_version_1103 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1103 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1103.bc")
add_library(oclc_isa_version_1150 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1150 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1150.bc")
add_library(oclc_isa_version_1151 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1151 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1151.bc")
add_library(oclc_isa_version_1152 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1152 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1152.bc")
add_library(oclc_isa_version_1153 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1153 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1153.bc")
add_library(oclc_isa_version_12-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_12-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_12-generic.bc")
add_library(oclc_isa_version_1200 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1200 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1200.bc")
add_library(oclc_isa_version_1201 STATIC IMPORTED)
set_target_properties(oclc_isa_version_1201 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1201.bc")
add_library(oclc_isa_version_600 STATIC IMPORTED)
set_target_properties(oclc_isa_version_600 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_600.bc")
add_library(oclc_isa_version_601 STATIC IMPORTED)
set_target_properties(oclc_isa_version_601 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_601.bc")
add_library(oclc_isa_version_602 STATIC IMPORTED)
set_target_properties(oclc_isa_version_602 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_602.bc")
add_library(oclc_isa_version_700 STATIC IMPORTED)
set_target_properties(oclc_isa_version_700 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_700.bc")
add_library(oclc_isa_version_701 STATIC IMPORTED)
set_target_properties(oclc_isa_version_701 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_701.bc")
add_library(oclc_isa_version_702 STATIC IMPORTED)
set_target_properties(oclc_isa_version_702 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_702.bc")
add_library(oclc_isa_version_703 STATIC IMPORTED)
set_target_properties(oclc_isa_version_703 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_703.bc")
add_library(oclc_isa_version_704 STATIC IMPORTED)
set_target_properties(oclc_isa_version_704 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_704.bc")
add_library(oclc_isa_version_705 STATIC IMPORTED)
set_target_properties(oclc_isa_version_705 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_705.bc")
add_library(oclc_isa_version_801 STATIC IMPORTED)
set_target_properties(oclc_isa_version_801 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_801.bc")
add_library(oclc_isa_version_802 STATIC IMPORTED)
set_target_properties(oclc_isa_version_802 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_802.bc")
add_library(oclc_isa_version_803 STATIC IMPORTED)
set_target_properties(oclc_isa_version_803 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_803.bc")
add_library(oclc_isa_version_805 STATIC IMPORTED)
set_target_properties(oclc_isa_version_805 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_805.bc")
add_library(oclc_isa_version_810 STATIC IMPORTED)
set_target_properties(oclc_isa_version_810 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_810.bc")
add_library(oclc_isa_version_9-4-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_9-4-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_9-4-generic.bc")
add_library(oclc_isa_version_9-generic STATIC IMPORTED)
set_target_properties(oclc_isa_version_9-generic PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_9-generic.bc")
add_library(oclc_isa_version_900 STATIC IMPORTED)
set_target_properties(oclc_isa_version_900 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_900.bc")
add_library(oclc_isa_version_902 STATIC IMPORTED)
set_target_properties(oclc_isa_version_902 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_902.bc")
add_library(oclc_isa_version_904 STATIC IMPORTED)
set_target_properties(oclc_isa_version_904 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_904.bc")
add_library(oclc_isa_version_906 STATIC IMPORTED)
set_target_properties(oclc_isa_version_906 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_906.bc")
add_library(oclc_isa_version_908 STATIC IMPORTED)
set_target_properties(oclc_isa_version_908 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_908.bc")
add_library(oclc_isa_version_909 STATIC IMPORTED)
set_target_properties(oclc_isa_version_909 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_909.bc")
add_library(oclc_isa_version_90a STATIC IMPORTED)
set_target_properties(oclc_isa_version_90a PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_90a.bc")
add_library(oclc_isa_version_90c STATIC IMPORTED)
set_target_properties(oclc_isa_version_90c PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_90c.bc")
add_library(oclc_isa_version_942 STATIC IMPORTED)
set_target_properties(oclc_isa_version_942 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_942.bc")
add_library(oclc_isa_version_950 STATIC IMPORTED)
set_target_properties(oclc_isa_version_950 PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_950.bc")
add_library(oclc_unsafe_math_off STATIC IMPORTED)
set_target_properties(oclc_unsafe_math_off PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_unsafe_math_off.bc")
add_library(oclc_unsafe_math_on STATIC IMPORTED)
set_target_properties(oclc_unsafe_math_on PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_unsafe_math_on.bc")
add_library(oclc_wavefrontsize64_off STATIC IMPORTED)
set_target_properties(oclc_wavefrontsize64_off PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_wavefrontsize64_off.bc")
add_library(oclc_wavefrontsize64_on STATIC IMPORTED)
set_target_properties(oclc_wavefrontsize64_on PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_wavefrontsize64_on.bc")
add_library(ocml STATIC IMPORTED)
set_target_properties(ocml PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/ocml.bc")
add_library(ockl STATIC IMPORTED)
set_target_properties(ockl PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/ockl.bc")
add_library(opencl STATIC IMPORTED)
set_target_properties(opencl PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/opencl.bc")
add_library(hip STATIC IMPORTED)
set_target_properties(hip PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/hip.bc")
add_library(asanrtl STATIC IMPORTED)
set_target_properties(asanrtl PROPERTIES
  IMPORTED_LOCATION "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/asanrtl.bc")

set_property(GLOBAL PROPERTY AMD_DEVICE_LIBS "oclc_abi_version_400;oclc_abi_version_500;oclc_abi_version_600;oclc_correctly_rounded_sqrt_off;oclc_correctly_rounded_sqrt_on;oclc_daz_opt_off;oclc_daz_opt_on;oclc_finite_only_off;oclc_finite_only_on;oclc_isa_version_10-1-generic;oclc_isa_version_10-3-generic;oclc_isa_version_1010;oclc_isa_version_1011;oclc_isa_version_1012;oclc_isa_version_1013;oclc_isa_version_1030;oclc_isa_version_1031;oclc_isa_version_1032;oclc_isa_version_1033;oclc_isa_version_1034;oclc_isa_version_1035;oclc_isa_version_1036;oclc_isa_version_11-generic;oclc_isa_version_1100;oclc_isa_version_1101;oclc_isa_version_1102;oclc_isa_version_1103;oclc_isa_version_1150;oclc_isa_version_1151;oclc_isa_version_1152;oclc_isa_version_1153;oclc_isa_version_12-generic;oclc_isa_version_1200;oclc_isa_version_1201;oclc_isa_version_600;oclc_isa_version_601;oclc_isa_version_602;oclc_isa_version_700;oclc_isa_version_701;oclc_isa_version_702;oclc_isa_version_703;oclc_isa_version_704;oclc_isa_version_705;oclc_isa_version_801;oclc_isa_version_802;oclc_isa_version_803;oclc_isa_version_805;oclc_isa_version_810;oclc_isa_version_9-4-generic;oclc_isa_version_9-generic;oclc_isa_version_900;oclc_isa_version_902;oclc_isa_version_904;oclc_isa_version_906;oclc_isa_version_908;oclc_isa_version_909;oclc_isa_version_90a;oclc_isa_version_90c;oclc_isa_version_942;oclc_isa_version_950;oclc_unsafe_math_off;oclc_unsafe_math_on;oclc_wavefrontsize64_off;oclc_wavefrontsize64_on;ocml;ockl;opencl;hip;asanrtl")

# List of exported target names.
set(AMD_DEVICE_LIBS_TARGETS "oclc_abi_version_400;oclc_abi_version_500;oclc_abi_version_600;oclc_correctly_rounded_sqrt_off;oclc_correctly_rounded_sqrt_on;oclc_daz_opt_off;oclc_daz_opt_on;oclc_finite_only_off;oclc_finite_only_on;oclc_isa_version_10-1-generic;oclc_isa_version_10-3-generic;oclc_isa_version_1010;oclc_isa_version_1011;oclc_isa_version_1012;oclc_isa_version_1013;oclc_isa_version_1030;oclc_isa_version_1031;oclc_isa_version_1032;oclc_isa_version_1033;oclc_isa_version_1034;oclc_isa_version_1035;oclc_isa_version_1036;oclc_isa_version_11-generic;oclc_isa_version_1100;oclc_isa_version_1101;oclc_isa_version_1102;oclc_isa_version_1103;oclc_isa_version_1150;oclc_isa_version_1151;oclc_isa_version_1152;oclc_isa_version_1153;oclc_isa_version_12-generic;oclc_isa_version_1200;oclc_isa_version_1201;oclc_isa_version_600;oclc_isa_version_601;oclc_isa_version_602;oclc_isa_version_700;oclc_isa_version_701;oclc_isa_version_702;oclc_isa_version_703;oclc_isa_version_704;oclc_isa_version_705;oclc_isa_version_801;oclc_isa_version_802;oclc_isa_version_803;oclc_isa_version_805;oclc_isa_version_810;oclc_isa_version_9-4-generic;oclc_isa_version_9-generic;oclc_isa_version_900;oclc_isa_version_902;oclc_isa_version_904;oclc_isa_version_906;oclc_isa_version_908;oclc_isa_version_909;oclc_isa_version_90a;oclc_isa_version_90c;oclc_isa_version_942;oclc_isa_version_950;oclc_unsafe_math_off;oclc_unsafe_math_on;oclc_wavefrontsize64_off;oclc_wavefrontsize64_on;ocml;ockl;opencl;hip;asanrtl")
