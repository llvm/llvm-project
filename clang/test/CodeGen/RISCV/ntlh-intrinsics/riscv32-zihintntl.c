// REQUIRES: riscv-registered-target
// RUN: %clang_cc1  -triple riscv32 -target-feature +v -target-feature +zihintntl -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <riscv_ntlh.h>
#include <riscv_vector.h>

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
vint32m1_t *scvi1, *scvi2;
vint16m1_t *scvs1, *scvs2;
vint8m1_t *scvc1, *scvc2;

// clang-format off
void ntl_all_sizes() {                                       // CHECK-LABEL: ntl_all_sizes
  uc = __riscv_ntl_load(&sc, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !7
  sc = __riscv_ntl_load(&uc, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !7
  us = __riscv_ntl_load(&ss, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  ss = __riscv_ntl_load(&us, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  ui = __riscv_ntl_load(&si, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  si = __riscv_ntl_load(&ui, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  ull = __riscv_ntl_load(&sll, __RISCV_NTLH_INNERMOST_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  sll = __riscv_ntl_load(&ull, __RISCV_NTLH_INNERMOST_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  h1 = __riscv_ntl_load(&h2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  f1 = __riscv_ntl_load(&f2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  d1 = __riscv_ntl_load(&d2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  v4si1 = __riscv_ntl_load(&v4si2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  v8ss1 = __riscv_ntl_load(&v8ss2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  v16sc1 = __riscv_ntl_load(&v16sc2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  *scvi1 = __riscv_ntl_load(scvi2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  *scvs1 = __riscv_ntl_load(scvs2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  *scvc1 = __riscv_ntl_load(scvc2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7

  uc = __riscv_ntl_load(&sc, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !8
  sc = __riscv_ntl_load(&uc, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !8
  us = __riscv_ntl_load(&ss, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  ss = __riscv_ntl_load(&us, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  ui = __riscv_ntl_load(&si, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  si = __riscv_ntl_load(&ui, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  ull = __riscv_ntl_load(&sll, __RISCV_NTLH_ALL_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  sll = __riscv_ntl_load(&ull, __RISCV_NTLH_ALL_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  h1 = __riscv_ntl_load(&h2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  f1 = __riscv_ntl_load(&f2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  d1 = __riscv_ntl_load(&d2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  v4si1 = __riscv_ntl_load(&v4si2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  v8ss1 = __riscv_ntl_load(&v8ss2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  v16sc1 = __riscv_ntl_load(&v16sc2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  *scvi1 = __riscv_ntl_load(scvi2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  *scvs1 = __riscv_ntl_load(scvs2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  *scvc1 = __riscv_ntl_load(scvc2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8

  uc = __riscv_ntl_load(&sc, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !9
  sc = __riscv_ntl_load(&uc, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !9
  us = __riscv_ntl_load(&ss, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  ss = __riscv_ntl_load(&us, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  ui = __riscv_ntl_load(&si, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  si = __riscv_ntl_load(&ui, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  ull = __riscv_ntl_load(&sll, __RISCV_NTLH_INNERMOST_SHARED); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  sll = __riscv_ntl_load(&ull, __RISCV_NTLH_INNERMOST_SHARED); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  h1 = __riscv_ntl_load(&h2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  f1 = __riscv_ntl_load(&f2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  d1 = __riscv_ntl_load(&d2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  v4si1 = __riscv_ntl_load(&v4si2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  v8ss1 = __riscv_ntl_load(&v8ss2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  v16sc1 = __riscv_ntl_load(&v16sc2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  *scvi1 = __riscv_ntl_load(scvi2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  *scvs1 = __riscv_ntl_load(scvs2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  *scvc1 = __riscv_ntl_load(scvc2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9

  uc = __riscv_ntl_load(&sc, __RISCV_NTLH_ALL);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  sc = __riscv_ntl_load(&uc, __RISCV_NTLH_ALL);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  us = __riscv_ntl_load(&ss, __RISCV_NTLH_ALL);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  ss = __riscv_ntl_load(&us, __RISCV_NTLH_ALL);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  ui = __riscv_ntl_load(&si, __RISCV_NTLH_ALL);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  si = __riscv_ntl_load(&ui, __RISCV_NTLH_ALL);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  ull = __riscv_ntl_load(&sll, __RISCV_NTLH_ALL); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  sll = __riscv_ntl_load(&ull, __RISCV_NTLH_ALL); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  h1 = __riscv_ntl_load(&h2, __RISCV_NTLH_ALL);   // CHECK: load half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  f1 = __riscv_ntl_load(&f2, __RISCV_NTLH_ALL);   // CHECK: load float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  d1 = __riscv_ntl_load(&d2, __RISCV_NTLH_ALL);   // CHECK: load double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  v4si1 = __riscv_ntl_load(&v4si2, __RISCV_NTLH_ALL);   // CHECK: load <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  v8ss1 = __riscv_ntl_load(&v8ss2, __RISCV_NTLH_ALL);   // CHECK: load <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  v16sc1 = __riscv_ntl_load(&v16sc2, __RISCV_NTLH_ALL);   // CHECK: load <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvi1 = __riscv_ntl_load(scvi2, __RISCV_NTLH_ALL);   // CHECK: load <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvs1 = __riscv_ntl_load(scvs2, __RISCV_NTLH_ALL);   // CHECK: load <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvc1 = __riscv_ntl_load(scvc2, __RISCV_NTLH_ALL);   // CHECK: load <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10

  uc = __riscv_ntl_load(&sc);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  sc = __riscv_ntl_load(&uc);   // CHECK: load i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  us = __riscv_ntl_load(&ss);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  ss = __riscv_ntl_load(&us);   // CHECK: load i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  ui = __riscv_ntl_load(&si);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  si = __riscv_ntl_load(&ui);   // CHECK: load i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  ull = __riscv_ntl_load(&sll); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  sll = __riscv_ntl_load(&ull); // CHECK: load i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  h1 = __riscv_ntl_load(&h2);   // CHECK: load half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  f1 = __riscv_ntl_load(&f2);   // CHECK: load float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  d1 = __riscv_ntl_load(&d2);   // CHECK: load double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  v4si1 = __riscv_ntl_load(&v4si2);   // CHECK: load <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  v8ss1 = __riscv_ntl_load(&v8ss2);   // CHECK: load <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  v16sc1 = __riscv_ntl_load(&v16sc2);   // CHECK: load <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvi1 = __riscv_ntl_load(scvi2);   // CHECK: load <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvs1 = __riscv_ntl_load(scvs2);   // CHECK: load <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  *scvc1 = __riscv_ntl_load(scvc2);   // CHECK: load <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10

  __riscv_ntl_store(&uc, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&sc, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&us, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&ss, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&ui, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&si, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&ull, 1, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&sll, 1, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&h1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&f1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&d1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&v4si1, v4si2, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&v8ss1, v8ss2, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(&v16sc1, v16sc2, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(scvi2, *scvi1, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(scvs2, *scvs1, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7
  __riscv_ntl_store(scvc2, *scvc1, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !7

  __riscv_ntl_store(&uc, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&sc, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&us, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&ss, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&ui, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&si, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&ull, 1, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&sll, 1, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&h1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&f1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&d1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&v4si1, v4si2, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&v8ss1, v8ss2, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(&v16sc1, v16sc2, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(scvi2, *scvi1, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(scvs2, *scvs1, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8
  __riscv_ntl_store(scvc2, *scvc1, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !8

  __riscv_ntl_store(&uc, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&sc, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&us, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&ss, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&ui, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&si, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&ull, 1, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&sll, 1, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&h1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&f1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&d1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&v4si1, v4si2, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&v8ss1, v8ss2, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(&v16sc1, v16sc2, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(scvi2, *scvi1, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(scvs2, *scvs1, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9
  __riscv_ntl_store(scvc2, *scvc1, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !9

  __riscv_ntl_store(&uc, 1, __RISCV_NTLH_ALL);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&sc, 1, __RISCV_NTLH_ALL);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&us, 1, __RISCV_NTLH_ALL);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ss, 1, __RISCV_NTLH_ALL);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ui, 1, __RISCV_NTLH_ALL);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&si, 1, __RISCV_NTLH_ALL);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ull, 1, __RISCV_NTLH_ALL);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&sll, 1, __RISCV_NTLH_ALL);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&h1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&f1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&d1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v4si1, v4si2, __RISCV_NTLH_ALL);  // CHECK: store <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v8ss1, v8ss2, __RISCV_NTLH_ALL);  // CHECK: store <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v16sc1, v16sc2, __RISCV_NTLH_ALL);  // CHECK: store <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvi2, *scvi1, __RISCV_NTLH_ALL);  // CHECK: store <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvs2, *scvs1, __RISCV_NTLH_ALL);  // CHECK: store <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvc2, *scvc1, __RISCV_NTLH_ALL);  // CHECK: store <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10

  __riscv_ntl_store(&uc, 1);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&sc, 1);    // CHECK: store i8{{.*}}align 1, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&us, 1);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ss, 1);    // CHECK: store i16{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ui, 1);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&si, 1);    // CHECK: store i32{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&ull, 1);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&sll, 1);   // CHECK: store i64{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&h1, 1.0);  // CHECK: store half{{.*}}align 2, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&f1, 1.0);  // CHECK: store float{{.*}}align 4, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&d1, 1.0);  // CHECK: store double{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v4si1, v4si2);  // CHECK: store <4 x i32>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v8ss1, v8ss2);  // CHECK: store <8 x i16>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(&v16sc1, v16sc2);  // CHECK: store <16 x i8>{{.*}}align 16, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvi2, *scvi1);  // CHECK: store <vscale x 2 x i32>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvs2, *scvs1);  // CHECK: store <vscale x 4 x i16>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
  __riscv_ntl_store(scvc2, *scvc1);  // CHECK: store <vscale x 8 x i8>{{.*}}align 8, !nontemporal !6, !riscv-nontemporal-domain !10
}
// clang-format on

// CHECK: !6 = !{i32 1}
// CHECK: !7 = !{i32 2}
// CHECK: !8 = !{i32 3}
// CHECK: !9 = !{i32 4}
// CHECK: !10 = !{i32 5}
