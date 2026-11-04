// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GNU
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=MSVC,BITINT
// RUN: %clang_cc1 -triple x86_64-windows-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23-GNU
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=COMPAT23-MSVC,BITINT

typedef int int128_t __attribute__((mode(TI)));

struct S {
  int128_t f();
};

int128_t S::f() { return 0; }

// GNU-LABEL: define dso_local void @_ZN1S1fEv(
// GNU-SAME: ptr dead_on_unwind noalias writable sret(i128) align 16 %agg.result,
// GNU-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this)
// MSVC-LABEL: define dso_local void @"?f@S@@QEAA_LXZ"(
// MSVC-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this,
// MSVC-SAME: ptr dead_on_unwind noalias writable sret(i128) align 16 %agg.result)
// COMPAT23-GNU-LABEL: define dso_local{{.*}} <2 x i64> @_ZN1S1fEv(
// COMPAT23-GNU-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this)
// COMPAT23-MSVC-LABEL: define dso_local{{.*}} <2 x i64> @"?f@S@@QEAA_LXZ"(
// COMPAT23-MSVC-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this)

struct BitInt {
  _BitInt(128) f();
};

_BitInt(128) BitInt::f() { return 0; }

// BITINT-LABEL: define dso_local void @"?f@BitInt
// BITINT-SAME: ptr dead_on_unwind noalias writable sret(i128) align 8 %agg.result,
// BITINT-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this)
