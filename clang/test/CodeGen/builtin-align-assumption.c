/// Check that the new alignment set by the alignment builtins is propagated
/// to e.g. llvm.memcpy calls.
// RUN: %clang_cc1 -triple=x86_64-unknown-unknown %s -emit-llvm -O1 -o - | FileCheck %s

// CHECK-LABEL: define {{[^@]+}}@align_up
// CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 64 dereferenceable(16) {{%.+}}, i8* nonnull align 1 dereferenceable(16) {{%.+}}, i64 16, i1 false)
// CHECK-NEXT:    ret void
//
void align_up(void* data, int* ptr) {
  // The call to llvm.memcpy should have an "align 64" on the first argument
  __builtin_memcpy(__builtin_align_up(ptr, 64), data, 16);
}
