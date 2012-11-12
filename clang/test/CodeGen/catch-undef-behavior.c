// RUN: %clang_cc1 -fsanitize=alignment,null,object-size,shift,return,signed-integer-overflow,vla-bound,float-cast-overflow,divide-by-zero -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fsanitize=null -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %clang_cc1 -fsanitize=signed-integer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-OVERFLOW

// CHECK: @[[INT:.*]] = private unnamed_addr constant { i16, i16, [6 x i8] } { i16 0, i16 11, [6 x i8] c"'int'\00" }

// FIXME: When we only emit each type once, use [[INT]] more below.
// CHECK: @[[LINE_100:.*]] = private unnamed_addr constant {{.*}}, i32 100, i32 5 {{.*}} @[[INT]], i64 4, i8 1
// CHECK: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 {{.*}}, i64 4, i8 0
// CHECK: @[[LINE_300_A:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_300_B:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 {{.*}} @{{.*}}, i64 4, i8 0 }
// CHECK: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 3 {{.*}} @{{.*}}, i64 4, i8 1 }

// CHECK: @[[STRUCT_S:.*]] = private unnamed_addr constant { i16, i16, [11 x i8] } { i16 -1, i16 0, [11 x i8] c"'struct S'\00" }

// CHECK: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 14 {{.*}} @[[STRUCT_S]], i64 4, i8 3 }
// CHECK: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 12 {{.*}} @{{.*}} }
// CHECK: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 11 {{.*}} @{{.*}} }

// CHECK-NULL: @[[LINE_100:.*]] = private unnamed_addr constant {{.*}}, i32 100, i32 5 {{.*}}

// PR6805
// CHECK: @foo
// CHECK-NULL: @foo
void foo() {
  union { int i; } u;
  // CHECK:      %[[CHECK0:.*]] = icmp ne {{.*}}* %[[PTR:.*]], null

  // CHECK:      %[[I8PTR:.*]] = bitcast i32* %[[PTR]] to i8*
  // CHECK-NEXT: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64(i8* %[[I8PTR]], i1 false)
  // CHECK-NEXT: %[[CHECK1:.*]] = icmp uge i64 %[[SIZE]], 4
  // CHECK-NEXT: %[[CHECK01:.*]] = and i1 %[[CHECK0]], %[[CHECK1]]

  // CHECK:      %[[PTRTOINT:.*]] = ptrtoint {{.*}}* %[[PTR]] to i64
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRTOINT]], 3
  // CHECK-NEXT: %[[CHECK2:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[OK:.*]] = and i1 %[[CHECK01]], %[[CHECK2]]
  // CHECK-NEXT: br i1 %[[OK]]

  // CHECK:      %[[ARG:.*]] = ptrtoint {{.*}} %[[PTR]] to i64
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %[[ARG]]) noreturn nounwind

  // With -fsanitize=null, only perform the null check.
  // CHECK-NULL: %[[NULL:.*]] = icmp ne {{.*}}, null
  // CHECK-NULL: br i1 %[[NULL]]
  // CHECK-NULL: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 100
  u.i=1;
}

// CHECK: @bar
int bar(int *a) {
  // CHECK:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK:      %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[ARG:.*]] = ptrtoint
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_200]] to i8*), i64 %[[ARG]]) noreturn nounwind
#line 200
  return *a;
}

// CHECK: @addr_space
int addr_space(int __attribute__((address_space(256))) *a) {
  // CHECK-NOT: __ubsan
  return *a;
}

// CHECK: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[INBOUNDS]]

  // FIXME: Only emit one trap block here.
  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300_A]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-NEXT: br i1 %[[NO_OVERFLOW]]

  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300_B]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
#line 300
  return a << b;
}

// CHECK: @rsh_inbounds
int rsh_inbounds(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ult i32 %[[RHS:.*]], 32
  // CHECK:      br i1 %[[INBOUNDS]]

  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_400]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[RET:.*]] = ashr i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
#line 400
  return a >> b;
}

// CHECK: @load
int load(int *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_500]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 500
  return *p;
}

// CHECK: @store
void store(int *p, int q) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_600]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 600
  *p = q;
}

struct S { int k; };

// CHECK: @member_access
int *member_access(struct S *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_700]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 700
  return &p->k;
}

// CHECK: @signed_overflow
int signed_overflow(int a, int b) {
  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_add_overflow(i8* bitcast ({{.*}} @[[LINE_800]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind
#line 800
  return a + b;
}

// CHECK: @no_return
int no_return() {
  // Reaching the end of a noreturn function is fine in C.
  // FIXME: If the user explicitly requests -fsanitize=return, we should catch
  //        that here even though it's not undefined behavior.
  // CHECK-NOT: call
  // CHECK-NOT: unreachable
  // CHECK: ret i32
}

// CHECK: @vla_bound
void vla_bound(int n) {
  // CHECK:      icmp sgt i32 %[[PARAM:.*]], 0
  //
  // CHECK:      %[[ARG:.*]] = zext i32 %[[PARAM]] to i64
  // CHECK-NEXT: call void @__ubsan_handle_vla_bound_not_positive(i8* bitcast ({{.*}} @[[LINE_900]] to i8*), i64 %[[ARG]]) noreturn nounwind
#line 900
  int arr[n * 3];
}

// CHECK: @int_float_no_overflow
float int_float_no_overflow(__int128 n) {
  // CHECK-NOT: call void @__ubsan_handle
  return n;
}

// CHECK: @int_float_overflow
float int_float_overflow(unsigned __int128 n) {
  // This is 2**104. FLT_MAX is 2**128 - 2**104.
  // CHECK: icmp ule i128 %{{.*}}, -20282409603651670423947251286016
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  return n;
}

// CHECK: @int_fp16_overflow
void int_fp16_overflow(int n, __fp16 *p) {
  // CHECK: %[[GE:.*]] = icmp sge i32 %{{.*}}, -65504
  // CHECK: %[[LE:.*]] = icmp sle i32 %{{.*}}, 65504
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  *p = n;
}

// CHECK: @float_int_overflow
int float_int_overflow(float f) {
  // CHECK: %[[GE:.*]] = fcmp oge float %[[F:.*]], 0xC1E0000000000000
  // CHECK: %[[LE:.*]] = fcmp ole float %[[F]], 0x41DFFFFFE0000000
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  return f;
}

// CHECK: @float_uint_overflow
unsigned float_uint_overflow(float f) {
  // CHECK: %[[GE:.*]] = fcmp oge float %[[F:.*]], 0.{{0*}}e+00
  // CHECK: %[[LE:.*]] = fcmp ole float %[[F]], 0x41EFFFFFE0000000
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  return f;
}

// CHECK: @fp16_char_overflow
signed char fp16_char_overflow(__fp16 *p) {
  // CHECK: %[[GE:.*]] = fcmp oge float %[[F:.*]], -1.28{{0*}}e+02
  // CHECK: %[[LE:.*]] = fcmp ole float %[[F]], 1.27{{0*}}e+02
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  return *p;
}

// CHECK: @float_float_overflow
float float_float_overflow(double f) {
  // CHECK: %[[GE:.*]] = fcmp oge double %[[F:.*]], 0xC7EFFFFFE0000000
  // CHECK: %[[LE:.*]] = fcmp ole double %[[F]], 0x47EFFFFFE0000000
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(
  return f;
}

// CHECK:          @int_divide_overflow
// CHECK-OVERFLOW: @int_divide_overflow
int int_divide_overflow(int a, int b) {
  // CHECK:               %[[ZERO:.*]] = icmp ne i32 %[[B:.*]], 0
  // CHECK-OVERFLOW-NOT:  icmp ne i32 %{{.*}}, 0

  // CHECK:               %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-NEXT:          %[[BOK:.*]] = icmp ne i32 %[[B]], -1
  // CHECK-NEXT:          %[[OVER:.*]] = or i1 %[[AOK]], %[[BOK]]

  // CHECK-OVERFLOW:      %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-OVERFLOW-NEXT: %[[BOK:.*]] = icmp ne i32 %[[B:.*]], -1
  // CHECK-OVERFLOW-NEXT: %[[OK:.*]] = or i1 %[[AOK]], %[[BOK]]

  // CHECK:               %[[OK:.*]] = and i1 %[[ZERO]], %[[OVER]]

  // CHECK:               br i1 %[[OK]]
  // CHECK-OVERFLOW:      br i1 %[[OK]]
  return a / b;

  // CHECK:          }
  // CHECK-OVERFLOW: }
}
