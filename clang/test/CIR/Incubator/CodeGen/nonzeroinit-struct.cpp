// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// TODO: Lower #cir.data_member<null> to -1 for LLVM (in the itanium ABI context).
// RUN-DISABLE: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN-DISABLE: FileCheck --input-file=%t.ll -check-prefix=LLVM %s

struct Other {
  int x;
};

struct Trivial {
  int x;
  double y;
  decltype(&Other::x) ptr;
};

// This case has a trivial default constructor, but can't be zero-initialized.
Trivial t;

// CHECK: !rec_Trivial = !cir.record<struct "Trivial" {!s32i, !cir.double, !cir.data_member<!s32i in !rec_Other>} #cir.record.decl.ast>
// CHECK: cir.global external @t = #cir.const_record<{#cir.int<0> : !s32i, #cir.fp<0.000000e+00> : !cir.double,
// CHECK-SAME: #cir.data_member<null> : !cir.data_member<!s32i in !rec_Other>}> : !rec_Trivial
