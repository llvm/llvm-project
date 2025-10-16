// RUN: %clang_cc1 -O -fexperimental-sanitize-metadata=atomics -triple x86_64-gnu-linux -x c -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,ATOMICS
// RUN: %clang_cc1 -O -fexperimental-sanitize-metadata=atomics -triple aarch64-gnu-linux -x c -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,ATOMICS

// CHECK: @__start_sanmd_atomics2 = extern_weak hidden global ptr
// CHECK: @__stop_sanmd_atomics2 = extern_weak hidden global ptr
// CHECK: @__start_sanmd_covered2 = extern_weak hidden global ptr
// CHECK: @__stop_sanmd_covered2 = extern_weak hidden global ptr

int x, y;

void empty() {
// CHECK-NOT: define dso_local void @empty() {{.*}} !pcsections
}

int atomics() {
// ATOMICS-LABEL: define dso_local i32 @atomics()
// ATOMICS-SAME:                                  !pcsections ![[ATOMICS_COVERED:[0-9]+]]
// ATOMICS-NEXT:  entry:
// ATOMICS-NEXT:    atomicrmw add {{.*}} !pcsections ![[ATOMIC_OP:[0-9]+]]
// ATOMICS-NOT:     load {{.*}} !pcsections
  __atomic_fetch_add(&x, 1, __ATOMIC_RELAXED);
  return y;
}
// ATOMICS-LABEL: __sanitizer_metadata_atomics2.module_ctor
// ATOMICS: call void @__sanitizer_metadata_atomics_add(i32 2, ptr @__start_sanmd_atomics2, ptr @__stop_sanmd_atomics2)
// ATOMICS-LABEL: __sanitizer_metadata_atomics2.module_dtor
// ATOMICS: call void @__sanitizer_metadata_atomics_del(i32 2, ptr @__start_sanmd_atomics2, ptr @__stop_sanmd_atomics2)

// CHECK-LABEL: __sanitizer_metadata_covered2.module_ctor
// CHECK: call void @__sanitizer_metadata_covered_add(i32 2, ptr @__start_sanmd_covered2, ptr @__stop_sanmd_covered2)
// CHECK-LABEL: __sanitizer_metadata_covered2.module_dtor
// CHECK: call void @__sanitizer_metadata_covered_del(i32 2, ptr @__start_sanmd_covered2, ptr @__stop_sanmd_covered2)

// ATOMICS: ![[ATOMICS_COVERED]] = !{!"sanmd_covered2!C", ![[ATOMICS_COVERED_AUX:[0-9]+]]}
// ATOMICS: ![[ATOMICS_COVERED_AUX]] = !{i64 1}
// ATOMICS: ![[ATOMIC_OP]] = !{!"sanmd_atomics2!C"}
