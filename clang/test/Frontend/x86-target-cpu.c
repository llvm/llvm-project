// Ensure we support the various CPU names.
//
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu nocona -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu core2 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu penryn -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu nehalem -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu westmere -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu sandybridge -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu ivybridge -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu haswell -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu broadwell -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu skylake -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu skylake-avx512 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu skx -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu cannonlake -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu icelake-client -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu icelake-server -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu knl -verify=knl %s
// knl-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// knl-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// knl-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu knm -verify=knm %s
// knm-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// knm-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// knm-warning@*:* {{KNL, KNM related Intel Xeon Phi CPU's specific ISA's supports will be removed in LLVM 19.}}
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu bonnell -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu silvermont -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu k8 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu opteron -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu athlon64 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu athlon-fx -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu k8-sse3 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu opteron-sse3 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu athlon64-sse3 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu amdfam10 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu barcelona -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu bdver1 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu bdver2 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu bdver3 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu bdver4 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu btver1 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu btver2 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu znver1 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu znver2 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu znver3 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-cpu znver4 -verify %s
//
// expected-no-diagnostics
