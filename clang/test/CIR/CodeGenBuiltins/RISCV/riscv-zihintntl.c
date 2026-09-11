// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintntl -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintntl -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintntl -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintntl -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintntl -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintntl -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

#include <riscv_ntlh.h>

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;
_Float16 h1, h2;
float f1, f2;
double d1, d2;
typedef int v4si __attribute__((vector_size(16)));
typedef signed short v8ss __attribute__((vector_size(16)));
typedef signed char v16sc __attribute__((vector_size(16)));
v4si v4si1, v4si2;
v8ss v8ss1, v8ss2;
v16sc v16sc1, v16sc2;

// CIR-LABEL: cir.func{{.*}} @ntl_load_all_sizes(
// LLVM-LABEL: @ntl_load_all_sizes(
void ntl_load_all_sizes(void) {
  // CIR: cir.load nontemporal align(1) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: load i8, ptr {{.*}}, align 1, !nontemporal [[NT:![0-9]+]], !riscv-nontemporal-domain [[D2:![0-9]+]]
  uc = __riscv_ntl_load(&sc, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.load nontemporal align(1) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: load i8, ptr {{.*}}, align 1, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3:![0-9]+]]
  sc = __riscv_ntl_load(&uc, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.load nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: load i16, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4:![0-9]+]]
  us = __riscv_ntl_load(&ss, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.load nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: load i16, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5:![0-9]+]]
  ss = __riscv_ntl_load(&us, __RISCV_NTLH_ALL);
  // CIR: cir.load nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: load i32, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  ui = __riscv_ntl_load(&si);
  // CIR: cir.load nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: load i32, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D2]]
  si = __riscv_ntl_load(&ui, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.load nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: load i64, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3]]
  ull = __riscv_ntl_load(&sll, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.load nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: load i64, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4]]
  sll = __riscv_ntl_load(&ull, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.load nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: load half, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  h1 = __riscv_ntl_load(&h2, __RISCV_NTLH_ALL);
  // CIR: cir.load nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: load float, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  f1 = __riscv_ntl_load(&f2);
  // CIR: cir.load nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: load double, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D2]]
  d1 = __riscv_ntl_load(&d2, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.load nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: load <4 x i32>, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3]]
  v4si1 = __riscv_ntl_load(&v4si2, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.load nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: load <8 x i16>, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4]]
  v8ss1 = __riscv_ntl_load(&v8ss2, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.load nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: load <16 x i8>, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  v16sc1 = __riscv_ntl_load(&v16sc2, __RISCV_NTLH_ALL);
}

// CIR-LABEL: cir.func{{.*}} @ntl_store_all_sizes(
// LLVM-LABEL: @ntl_store_all_sizes(
void ntl_store_all_sizes(void) {
  // CIR: cir.store nontemporal align(1) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: store i8 {{.*}}, ptr {{.*}}, align 1, !nontemporal [[NT]], !riscv-nontemporal-domain [[D2]]
  __riscv_ntl_store(&uc, 1, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.store nontemporal align(1) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: store i8 {{.*}}, ptr {{.*}}, align 1, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3]]
  __riscv_ntl_store(&sc, 1, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.store nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: store i16 {{.*}}, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4]]
  __riscv_ntl_store(&us, 1, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.store nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: store i16 {{.*}}, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  __riscv_ntl_store(&ss, 1, __RISCV_NTLH_ALL);
  // CIR: cir.store nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: store i32 {{.*}}, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  __riscv_ntl_store(&ui, 1);
  // CIR: cir.store nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: store i32 {{.*}}, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D2]]
  __riscv_ntl_store(&si, 1, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.store nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: store i64 {{.*}}, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3]]
  __riscv_ntl_store(&ull, 1, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.store nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: store i64 {{.*}}, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4]]
  __riscv_ntl_store(&sll, 1, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.store nontemporal align(2) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: store half {{.*}}, ptr {{.*}}, align 2, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  __riscv_ntl_store(&h1, 1.0, __RISCV_NTLH_ALL);
  // CIR: cir.store nontemporal align(4) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: store float {{.*}}, ptr {{.*}}, align 4, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  __riscv_ntl_store(&f1, 1.0);
  // CIR: cir.store nontemporal align(8) {{.*}} {cir.riscv_nontemporal_domain = 2 : i32}
  // LLVM: store double {{.*}}, ptr {{.*}}, align 8, !nontemporal [[NT]], !riscv-nontemporal-domain [[D2]]
  __riscv_ntl_store(&d1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);
  // CIR: cir.store nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 3 : i32}
  // LLVM: store <4 x i32> {{.*}}, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D3]]
  __riscv_ntl_store(&v4si1, v4si2, __RISCV_NTLH_ALL_PRIVATE);
  // CIR: cir.store nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 4 : i32}
  // LLVM: store <8 x i16> {{.*}}, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D4]]
  __riscv_ntl_store(&v8ss1, v8ss2, __RISCV_NTLH_INNERMOST_SHARED);
  // CIR: cir.store nontemporal align(16) {{.*}} {cir.riscv_nontemporal_domain = 5 : i32}
  // LLVM: store <16 x i8> {{.*}}, ptr {{.*}}, align 16, !nontemporal [[NT]], !riscv-nontemporal-domain [[D5]]
  __riscv_ntl_store(&v16sc1, v16sc2, __RISCV_NTLH_ALL);
}

// LLVM-DAG: [[NT]] = !{i32 1}
// LLVM-DAG: [[D2]] = !{i32 2}
// LLVM-DAG: [[D3]] = !{i32 3}
// LLVM-DAG: [[D4]] = !{i32 4}
// LLVM-DAG: [[D5]] = !{i32 5}
