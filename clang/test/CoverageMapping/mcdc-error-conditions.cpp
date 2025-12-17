// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND7,TV

// RUN: %clang_cc1 -fmcdc-max-test-vectors=8 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND7,TV
// RUN: %clang_cc1 -fmcdc-max-test-vectors=7 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,TV7,TV

// RUN: %clang_cc1 -fmcdc-max-conditions=287 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND7,TV
// RUN: %clang_cc1 -fmcdc-max-conditions=286 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND7,COND
// RUN: %clang_cc1 -fmcdc-max-conditions=7   -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND7,COND
// RUN: %clang_cc1 -fmcdc-max-conditions=6   -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s --check-prefixes=CHECK,COND6,COND

bool func_conditions(bool a, bool b, bool c, bool d, bool e, bool f, bool g) {
  //   TV7: :[[@LINE+2]]:10: warning: unsupported MC/DC boolean expression; number of test vectors (8) exceeds max
  // COND6: :[[@LINE+1]]:10: warning: unsupported MC/DC boolean expression; number of conditions (7) exceeds max
  return a && b && c && d && e && f && g;
}

// From clang-tidy/misc/MisleadingIdentifier.cpp
bool func_isR(unsigned CP) {
  //   TV: :[[@LINE+2]]:10: warning: unsupported MC/DC boolean expression; number of test vectors (2147483647) exceeds max
  // COND: :[[@LINE+1]]:10: warning: unsupported MC/DC boolean expression; number of conditions (287) exceeds max
  return (CP == 0x0590) || (CP == 0x05BE) || (CP == 0x05C0) || (CP == 0x05C3) ||
         (CP == 0x05C6) || (0x05C8 <= CP && CP <= 0x05CF) ||
         (0x05D0 <= CP && CP <= 0x05EA) || (0x05EB <= CP && CP <= 0x05EE) ||
         (0x05EF <= CP && CP <= 0x05F2) || (0x05F3 <= CP && CP <= 0x05F4) ||
         (0x05F5 <= CP && CP <= 0x05FF) || (0x07C0 <= CP && CP <= 0x07C9) ||
         (0x07CA <= CP && CP <= 0x07EA) || (0x07F4 <= CP && CP <= 0x07F5) ||
         (CP == 0x07FA) || (0x07FB <= CP && CP <= 0x07FC) ||
         (0x07FE <= CP && CP <= 0x07FF) || (0x0800 <= CP && CP <= 0x0815) ||
         (CP == 0x081A) || (CP == 0x0824) || (CP == 0x0828) ||
         (0x082E <= CP && CP <= 0x082F) || (0x0830 <= CP && CP <= 0x083E) ||
         (CP == 0x083F) || (0x0840 <= CP && CP <= 0x0858) ||
         (0x085C <= CP && CP <= 0x085D) || (CP == 0x085E) || (CP == 0x085F) ||
         (CP == 0x200F) || (CP == 0xFB1D) || (0xFB1F <= CP && CP <= 0xFB28) ||
         (0xFB2A <= CP && CP <= 0xFB36) || (CP == 0xFB37) ||
         (0xFB38 <= CP && CP <= 0xFB3C) || (CP == 0xFB3D) || (CP == 0xFB3E) ||
         (CP == 0xFB3F) || (0xFB40 <= CP && CP <= 0xFB41) || (CP == 0xFB42) ||
         (0xFB43 <= CP && CP <= 0xFB44) || (CP == 0xFB45) ||
         (0xFB46 <= CP && CP <= 0xFB4F) || (0x10800 <= CP && CP <= 0x10805) ||
         (0x10806 <= CP && CP <= 0x10807) || (CP == 0x10808) ||
         (CP == 0x10809) || (0x1080A <= CP && CP <= 0x10835) ||
         (CP == 0x10836) || (0x10837 <= CP && CP <= 0x10838) ||
         (0x10839 <= CP && CP <= 0x1083B) || (CP == 0x1083C) ||
         (0x1083D <= CP && CP <= 0x1083E) || (0x1083F <= CP && CP <= 0x10855) ||
         (CP == 0x10856) || (CP == 0x10857) ||
         (0x10858 <= CP && CP <= 0x1085F) || (0x10860 <= CP && CP <= 0x10876) ||
         (0x10877 <= CP && CP <= 0x10878) || (0x10879 <= CP && CP <= 0x1087F) ||
         (0x10880 <= CP && CP <= 0x1089E) || (0x1089F <= CP && CP <= 0x108A6) ||
         (0x108A7 <= CP && CP <= 0x108AF) || (0x108B0 <= CP && CP <= 0x108DF) ||
         (0x108E0 <= CP && CP <= 0x108F2) || (CP == 0x108F3) ||
         (0x108F4 <= CP && CP <= 0x108F5) || (0x108F6 <= CP && CP <= 0x108FA) ||
         (0x108FB <= CP && CP <= 0x108FF) || (0x10900 <= CP && CP <= 0x10915) ||
         (0x10916 <= CP && CP <= 0x1091B) || (0x1091C <= CP && CP <= 0x1091E) ||
         (0x10920 <= CP && CP <= 0x10939) || (0x1093A <= CP && CP <= 0x1093E) ||
         (CP == 0x1093F) || (0x10940 <= CP && CP <= 0x1097F) ||
         (0x10980 <= CP && CP <= 0x109B7) || (0x109B8 <= CP && CP <= 0x109BB) ||
         (0x109BC <= CP && CP <= 0x109BD) || (0x109BE <= CP && CP <= 0x109BF) ||
         (0x109C0 <= CP && CP <= 0x109CF) || (0x109D0 <= CP && CP <= 0x109D1) ||
         (0x109D2 <= CP && CP <= 0x109FF) || (CP == 0x10A00) ||
         (CP == 0x10A04) || (0x10A07 <= CP && CP <= 0x10A0B) ||
         (0x10A10 <= CP && CP <= 0x10A13) || (CP == 0x10A14) ||
         (0x10A15 <= CP && CP <= 0x10A17) || (CP == 0x10A18) ||
         (0x10A19 <= CP && CP <= 0x10A35) || (0x10A36 <= CP && CP <= 0x10A37) ||
         (0x10A3B <= CP && CP <= 0x10A3E) || (0x10A40 <= CP && CP <= 0x10A48) ||
         (0x10A49 <= CP && CP <= 0x10A4F) || (0x10A50 <= CP && CP <= 0x10A58) ||
         (0x10A59 <= CP && CP <= 0x10A5F) || (0x10A60 <= CP && CP <= 0x10A7C) ||
         (0x10A7D <= CP && CP <= 0x10A7E) || (CP == 0x10A7F) ||
         (0x10A80 <= CP && CP <= 0x10A9C) || (0x10A9D <= CP && CP <= 0x10A9F) ||
         (0x10AA0 <= CP && CP <= 0x10ABF) || (0x10AC0 <= CP && CP <= 0x10AC7) ||
         (CP == 0x10AC8) || (0x10AC9 <= CP && CP <= 0x10AE4) ||
         (0x10AE7 <= CP && CP <= 0x10AEA) || (0x10AEB <= CP && CP <= 0x10AEF) ||
         (0x10AF0 <= CP && CP <= 0x10AF6) || (0x10AF7 <= CP && CP <= 0x10AFF) ||
         (0x10B00 <= CP && CP <= 0x10B35) || (0x10B36 <= CP && CP <= 0x10B38) ||
         (0x10B40 <= CP && CP <= 0x10B55) || (0x10B56 <= CP && CP <= 0x10B57) ||
         (0x10B58 <= CP && CP <= 0x10B5F) || (0x10B60 <= CP && CP <= 0x10B72) ||
         (0x10B73 <= CP && CP <= 0x10B77) || (0x10B78 <= CP && CP <= 0x10B7F) ||
         (0x10B80 <= CP && CP <= 0x10B91) || (0x10B92 <= CP && CP <= 0x10B98) ||
         (0x10B99 <= CP && CP <= 0x10B9C) || (0x10B9D <= CP && CP <= 0x10BA8) ||
         (0x10BA9 <= CP && CP <= 0x10BAF) || (0x10BB0 <= CP && CP <= 0x10BFF) ||
         (0x10C00 <= CP && CP <= 0x10C48) || (0x10C49 <= CP && CP <= 0x10C7F) ||
         (0x10C80 <= CP && CP <= 0x10CB2) || (0x10CB3 <= CP && CP <= 0x10CBF) ||
         (0x10CC0 <= CP && CP <= 0x10CF2) || (0x10CF3 <= CP && CP <= 0x10CF9) ||
         (0x10CFA <= CP && CP <= 0x10CFF) || (0x10D40 <= CP && CP <= 0x10E5F) ||
         (CP == 0x10E7F) || (0x10E80 <= CP && CP <= 0x10EA9) ||
         (CP == 0x10EAA) || (CP == 0x10EAD) ||
         (0x10EAE <= CP && CP <= 0x10EAF) || (0x10EB0 <= CP && CP <= 0x10EB1) ||
         (0x10EB2 <= CP && CP <= 0x10EFF) || (0x10F00 <= CP && CP <= 0x10F1C) ||
         (0x10F1D <= CP && CP <= 0x10F26) || (CP == 0x10F27) ||
         (0x10F28 <= CP && CP <= 0x10F2F) || (0x10F70 <= CP && CP <= 0x10F81) ||
         (0x10F86 <= CP && CP <= 0x10F89) || (0x10F8A <= CP && CP <= 0x10FAF) ||
         (0x10FB0 <= CP && CP <= 0x10FC4) || (0x10FC5 <= CP && CP <= 0x10FCB) ||
         (0x10FCC <= CP && CP <= 0x10FDF) || (0x10FE0 <= CP && CP <= 0x10FF6) ||
         (0x10FF7 <= CP && CP <= 0x10FFF) || (0x1E800 <= CP && CP <= 0x1E8C4) ||
         (0x1E8C5 <= CP && CP <= 0x1E8C6) || (0x1E8C7 <= CP && CP <= 0x1E8CF) ||
         (0x1E8D7 <= CP && CP <= 0x1E8FF) || (0x1E900 <= CP && CP <= 0x1E943) ||
         (CP == 0x1E94B) || (0x1E94C <= CP && CP <= 0x1E94F) ||
         (0x1E950 <= CP && CP <= 0x1E959) || (0x1E95A <= CP && CP <= 0x1E95D) ||
         (0x1E95E <= CP && CP <= 0x1E95F) || (0x1E960 <= CP && CP <= 0x1EC6F) ||
         (0x1ECC0 <= CP && CP <= 0x1ECFF) || (0x1ED50 <= CP && CP <= 0x1EDFF);
}

// CHECK: _Z15func_conditionsbbbbbbb:
//   TV8-NOT: Decision,
// COND6-NOT: Decision,
//     COND7: Decision,File 0, {{[0-9]+}}:10 -> {{[0-9]+}}:41 = M:8, C:7
// CHECK: _Z8func_isRj:
// CHECK-NOT: Decision,
// CHECK-NOT: Branch,{{.*}}]
//     CHECK: Branch,File 0, [[@LINE-10]]:64 -> [[@LINE-10]]:77 =
