/// Normally -g1 does not add linkageName. -fdebug-info-for-profiling adds linkageName.
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=LINE
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -debug-info-kind=line-tables-only -fdebug-info-for-profiling %s -o - | FileCheck %s

// LINE: = distinct !DISubprogram(name: "foo", scope:

// CHECK: = distinct !DICompileUnit({{.*}}, debugInfoForProfiling: true,
// CHECK: = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope:

/// Add a DWARF discriminators pass for PGO.
// RUN: %clang_cc1 -emit-llvm -fdebug-pass-manager -O1 -fprofile-instrument-path=a.profdata %s -o - 2>&1 | FileCheck %s --check-prefix=NODISCR
// RUN: %clang_cc1 -emit-llvm -fdebug-pass-manager -O1 -fprofile-instrument-path=a.profdata -fdebug-info-for-profiling %s -o - 2>&1 | FileCheck %s --check-prefix=DISCR

// RUN: echo > %t.proftext
// RUN: llvm-profdata merge %t.proftext -o %t.profdata
// RUN: %clang_cc1 -emit-llvm -fdebug-pass-manager -O1 -fprofile-instrument-use-path=%t.profdata -fdebug-info-for-profiling %s -o - 2>&1 | FileCheck %s --check-prefix=DISCR
// RUN: %clang_cc1 -emit-llvm -fdebug-pass-manager -O1 -fdebug-info-for-profiling -fpseudo-probe-for-profiling %s -o - 2>&1 | FileCheck %s --check-prefix=PROBE

// NODISCR-NOT: Running pass: AddDiscriminatorsPass
// DISCR:       Running pass: AddDiscriminatorsPass on {{.*}}
// PROBE:       Running pass: AddDiscriminatorsPass on {{.*}}
// PROBE:       Running pass: SampleProfileProbePass on {{.*}}

void foo() {}
