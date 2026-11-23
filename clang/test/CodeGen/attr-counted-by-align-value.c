// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK,NOSAN %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=array-bounds -emit-llvm -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK,BOUNDS %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=array-bounds,alignment -emit-llvm -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK,BOUNDS,ALIGN %s

// With -fsanitize=array-bounds, a subscript of a 'counted_by' pointer field
// loads the pointer itself rather than going through the scalar emitter, so
// that it can reuse the field's address for the bounds check. Make sure the
// 'align_value' assumption on the field's type (and its
// -fsanitize=alignment check) is still emitted on that path.

typedef int *__attribute__((align_value(64))) aligned_int_ptr;

struct s {
  int n;
  aligned_int_ptr buf __attribute__((counted_by(n)));
};

// CHECK-LABEL: define {{.*}} @f(
// CHECK:         [[FIELD:%.*]] = getelementptr inbounds nuw %struct.s, ptr {{%.*}}, i32 0, i32 1
// CHECK:         [[BUF:%.*]] = load ptr, ptr [[FIELD]]
// ALIGN:         call void @__ubsan_handle_alignment_assumption
// CHECK:         call void @llvm.assume(i1 true) [ "align"(ptr [[BUF]], i64 64) ]
// CHECK:         getelementptr inbounds i32, ptr [[BUF]]
// BOUNDS:        getelementptr inbounds i8, ptr [[FIELD]], i32 -8
// BOUNDS:        call void @__ubsan_handle_out_of_bounds
// NOSAN-NOT:     __ubsan_handle
int f(struct s *p, int i) { return p->buf[i]; }
