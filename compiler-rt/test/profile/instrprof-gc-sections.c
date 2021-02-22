// REQUIRES: linux, lld-available

// RUN: %clang_profgen=%t.profraw -fuse-ld=lld -fcoverage-mapping -mllvm -enable-name-compression=false -DCODE=1 -ffunction-sections -fdata-sections -Wl,--gc-sections -o %t %s
// RUN: %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --all-functions %t.profdata | FileCheck %s -check-prefix=PROF
// RUN: llvm-cov show %t -instr-profile %t.profdata | FileCheck %s -check-prefix=COV
// RUN: llvm-nm %t | FileCheck %s -check-prefix=NM
// RUN: llvm-readelf -x __llvm_prf_names %t | FileCheck %s -check-prefix=PRF_NAMES
// RUN: llvm-size -A %t | FileCheck %s -check-prefix=PRF_CNTS

// RUN: %clang_lto_profgen=%t.lto.profraw -fuse-ld=lld -fcoverage-mapping -mllvm -enable-name-compression=false -DCODE=1 -ffunction-sections -fdata-sections -Wl,--gc-sections -flto -o %t.lto %s
// RUN: %run %t.lto
// RUN: llvm-profdata merge -o %t.lto.profdata %t.lto.profraw
// RUN: llvm-profdata show --all-functions %t.lto.profdata | FileCheck %s -check-prefix=PROF
// RUN: llvm-cov show %t.lto -instr-profile %t.lto.profdata | FileCheck %s -check-prefix=COV
// RUN: llvm-nm %t.lto | FileCheck %s -check-prefix=NM
// RUN: llvm-readelf -x __llvm_prf_names %t.lto | FileCheck %s -check-prefix=PRF_NAMES
// RUN: llvm-size -A %t.lto | FileCheck %s -check-prefix=PRF_CNTS

// Note: We expect foo() and some of the profiling data associated with it to
// be garbage collected.

// Note: When there is no code in a program, we expect to see the exact same
// set of external functions provided by the profile runtime.

// Note: We also check the IR instrumentation and expect foo() to be garbage
// collected as well.

// RUN: %clang_pgogen=%t.pgo.profraw -fuse-ld=lld -DCODE=1 -ffunction-sections -fdata-sections -Wl,--gc-sections -o %t.pgo %s
// RUN: %run %t.pgo
// RUN: llvm-profdata merge -o %t.pgo.profdata %t.pgo.profraw
// RUN: llvm-profdata show --all-functions %t.pgo.profdata | FileCheck %s -check-prefix=PGO
// RUN: llvm-nm %t.pgo | FileCheck %s -check-prefix=NM

#ifdef CODE

// COV: [[@LINE+1]]{{ *}}|{{ *}}0|void foo()
void foo() {}

// COV: [[@LINE+1]]{{ *}}|{{ *}}1|int main
int main() { return 0; }

#endif // CODE

// NM-NOT: foo

// PROF: Counters:
// PROF-NEXT:   main:
// PROF-NEXT:     Hash:
// PROF-NEXT:     Counters: 1
// PROF-NEXT:     Function count: 1
// PROF-NEXT: Instrumentation level: Front-end
// PROF-NEXT: Functions shown: 1
// PROF-NEXT: Total functions: 1
// PROF-NEXT: Maximum function count:
// PROF-NEXT: Maximum internal block count:

// Note: We don't expect the names of garbage collected functions to disappear
// from __llvm_prf_names, because collectPGOFuncNameStrings() glues the names
// together.

// PRF_NAMES: Hex dump of section '__llvm_prf_names':
// PRF_NAMES-NEXT: {{.*}} 0800666f 6f016d61 696e{{.*$}}
//                        | | f o  o # m a  i n
//                        | |___________|
//                        |             |
//               UncompressedLen = 8    |
//                                      |
//                               CompressedLen = 0

// Note: We expect the profile counters for garbage collected functions to also
// be garbage collected.

// PRF_CNTS: __llvm_prf_cnts 8

// PGO: Counters:
// PGO-NEXT:   main:
// PGO-NEXT:     Hash:
// PGO-NEXT:     Counters: 1
// PGO-NEXT: Instrumentation level: IR
// PGO-NEXT: Functions shown: 1
// PGO-NEXT: Total functions: 1
// PGO-NEXT: Maximum function count:
// PGO-NEXT: Maximum internal block count:
