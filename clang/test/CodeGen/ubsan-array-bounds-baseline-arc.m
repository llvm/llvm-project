// Baseline for the Objective-C ARC contexts a subscript can appear in. Split
// from ubsan-array-bounds-baseline.c because ARC's __weak needs a Darwin
// triple, not because the contexts are unrelated.
// RUN: %clang_cc1 -triple arm64-apple-macosx11.0.0 -emit-llvm \
// RUN:     -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -fobjc-arc -fobjc-runtime-has-weak -fblocks \
// RUN:     %s -o - | FileCheck %s

__weak id wa[4];
id st[4];
__unsafe_unretained id ua[4];
__strong id *p;

//===----------------------------------------------------------------------===//
// Contexts that require the element to exist.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@arc_weak_store(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_weak_store(int i, id v) { wa[i] = v; }

// CHECK-LABEL: define {{.*}}@arc_strong_store(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_strong_store(int i, id v) { st[i] = v; }

// CHECK-LABEL: define {{.*}}@arc_weak_load(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_weak_load(int i, id *o) { *o = wa[i]; }

// The remaining lifetimes take their own emitters, so each is covered rather
// than assumed to follow from the two above.

// CHECK-LABEL: define {{.*}}@arc_strong_load(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_strong_load(int i, id *o) { *o = st[i]; }

// CHECK-LABEL: define {{.*}}@arc_unsafe_store(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_unsafe_store(int i, id v) { ua[i] = v; }

// CHECK-LABEL: define {{.*}}@arc_unsafe_load(
// CHECK: icmp ult i64 {{.*}}, 4
void arc_unsafe_load(int i, id *o) { *o = ua[i]; }

//===----------------------------------------------------------------------===//
// Address-only: `ule` is required.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@arc_ctl_addr(
// CHECK: icmp ule i64 {{.*}}, 4
void arc_ctl_addr(int i) { p = &st[i]; }
