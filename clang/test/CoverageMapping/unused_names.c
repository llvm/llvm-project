// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -main-file-name unused_names.c -o - %s > %t
// RUN: FileCheck -input-file %t %s
// RUN: FileCheck -check-prefix=SYSHEADER -input-file %t %s

// CHECK-DAG: @__profc_bar
// CHECK-DAG: @__llvm_prf_nm = private constant {{.*}}, section "{{.*__llvm_prf_names|.*lprfn}}"

// These are never instantiated, so we shouldn't get counters for them.
//
// CHECK-NOT: @__profc_baz
// CHECK-NOT: @__profc_unused_names.c_qux

// SYSHEADER-NOT: @__profc_foo =


#ifdef IS_SYSHEADER

#pragma clang system_header
inline int foo() { return 0; }

#else

#define IS_SYSHEADER
#include __FILE__

int bar() { return 0; }
inline int baz() { return 0; }
static int qux() { return 42; }

#endif
