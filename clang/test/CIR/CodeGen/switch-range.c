// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

typedef unsigned long long u64;

// Large unsigned range; high-bit upper bound must not be treated as empty.
int range_u64(u64 x) {
  switch (x) {
  case 1000000000000000000ULL ... 9999999999999999999ULL:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_u64
// CIR: cir.case(range, [#cir.int<1000000000000000000> : !u64i, #cir.int<9999999999999999999> : !u64i])

// LLVM-LABEL: define dso_local i32 @range_u64(
// LLVM:        %[[D:.*]] = sub i64 %{{.*}}, 1000000000000000000
// LLVM-NEXT:   %[[C:.*]] = icmp ule i64 %[[D]], 8999999999999999999
// LLVM-NEXT:   br i1 %[[C]], label %{{.*}}, label %{{.*}}

// Small unsigned range at the top of the domain.
int range_u64_top(u64 x) {
  switch (x) {
  case 18446744073709551613ULL ... 18446744073709551615ULL:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_u64_top
// CIR: cir.case(range, [#cir.int<18446744073709551613> : !u64i, #cir.int<18446744073709551615> : !u64i])

// LLVM-LABEL: define dso_local i32 @range_u64_top(
// LLVM:        switch i64 %{{.*}}, label %{{.*}} [
// LLVM-NEXT:     i64 -3, label %[[BB:.*]]
// LLVM-NEXT:     i64 -2, label %[[BB]]
// LLVM-NEXT:     i64 -1, label %[[BB]]
// LLVM-NEXT:   ]

// Signed range spanning negative to positive, wider than the expansion
// threshold.
int range_s32_neg(int x) {
  switch (x) {
  case -2000000000 ... 2000000000:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_s32_neg
// CIR: cir.case(range, [#cir.int<-2000000000> : !s32i, #cir.int<2000000000> : !s32i])

// LLVM-LABEL: define dso_local i32 @range_s32_neg(
// LLVM:        %[[D:.*]] = sub i32 %{{.*}}, -2000000000
// LLVM-NEXT:   %[[C:.*]] = icmp ule i32 %[[D]], -294967296
// LLVM-NEXT:   br i1 %[[C]], label %{{.*}}, label %{{.*}}

// Small signed range (existing expansion path).
int range_s32_small(int x) {
  switch (x) {
  case 3 ... 6:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_s32_small
// CIR: cir.case(range, [#cir.int<3> : !s32i, #cir.int<6> : !s32i])

// LLVM-LABEL: define dso_local i32 @range_s32_small(
// LLVM:        switch i32 %{{.*}}, label %{{.*}} [
// LLVM-NEXT:     i32 3, label %[[BB:.*]]
// LLVM-NEXT:     i32 4, label %[[BB]]
// LLVM-NEXT:     i32 5, label %[[BB]]
// LLVM-NEXT:     i32 6, label %[[BB]]
// LLVM-NEXT:   ]

// Small signed range at the top of the domain: the expansion cursor must not
// run past INT_MAX.
int range_s32_top(int x) {
  switch (x) {
  case 2147483645 ... 2147483647:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_s32_top
// CIR: cir.case(range, [#cir.int<2147483645> : !s32i, #cir.int<2147483647> : !s32i])

// LLVM-LABEL: define dso_local i32 @range_s32_top(
// LLVM:        switch i32 %{{.*}}, label %{{.*}} [
// LLVM-NEXT:     i32 2147483645, label %[[BB:.*]]
// LLVM-NEXT:     i32 2147483646, label %[[BB]]
// LLVM-NEXT:     i32 2147483647, label %[[BB]]
// LLVM-NEXT:   ]

// A range of size 64 sits just past the expansion threshold, so it lowers via
// the sub + ule range check rather than individual cases.
int range_thresh64(int x) {
  switch (x) {
  case 0 ... 64:
    return 1;
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @range_thresh64
// CIR: cir.case(range, [#cir.int<0> : !s32i, #cir.int<64> : !s32i])

// LLVM-LABEL: define dso_local i32 @range_thresh64(
// LLVM:        %[[D:.*]] = sub i32 %{{.*}}, 0
// LLVM-NEXT:   %[[C:.*]] = icmp ule i32 %[[D]], 64
// LLVM-NEXT:   br i1 %[[C]], label %{{.*}}, label %{{.*}}
