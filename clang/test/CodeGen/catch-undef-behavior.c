// RUN: %clang_cc1 -fsanitize=alignment,null,object-size,shift-base,shift-exponent,return,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -fsanitize-recover=alignment,null,object-size,shift-base,shift-exponent,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -emit-llvm %s -o - -triple x86_64-linux-gnu | opt -instnamer -S | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-UBSAN
// RUN: %clang_cc1 -fsanitize-trap=alignment,null,object-size,shift-base,shift-exponent,return,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -fsanitize-recover=alignment,null,object-size,shift-base,shift-exponent,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -fsanitize=alignment,null,object-size,shift-base,shift-exponent,return,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -fsanitize-recover=alignment,null,object-size,shift-base,shift-exponent,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -emit-llvm %s -o - -triple x86_64-linux-gnu | opt -instnamer -S | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-TRAP
// RUN: %clang_cc1 -fsanitize=signed-integer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-OVERFLOW

// CHECK-UBSAN: @[[INT:.*]] = private unnamed_addr constant { i16, i16, [6 x i8] } { i16 0, i16 11, [6 x i8] c"'int'\00" }

// FIXME: When we only emit each type once, use [[INT]] more below.
// CHECK-UBSAN: @[[LINE_100:.*]] = private unnamed_addr global {{.*}}, i32 100, i32 5 {{.*}} @[[INT]], i8 2, i8 1
// CHECK-UBSAN: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 {{.*}}, i8 2, i8 0
// CHECK-UBSAN: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK-UBSAN: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK-UBSAN: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 {{.*}} @{{.*}}, i8 2, i8 0 }
// CHECK-UBSAN: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 3 {{.*}} @{{.*}}, i8 2, i8 1 }

// CHECK-UBSAN: @[[STRUCT_S:.*]] = private unnamed_addr constant { i16, i16, [11 x i8] } { i16 -1, i16 0, [11 x i8] c"'struct S'\00" }

// CHECK-UBSAN: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 14 {{.*}} @[[STRUCT_S]], i8 2, i8 3 }
// CHECK-UBSAN: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 12 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 11 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[LINE_1000:.*]] = {{.*}}, i32 1000, i32 10 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[FP16:.*]] = private unnamed_addr constant { i16, i16, [9 x i8] } { i16 1, i16 16, [9 x i8] c"'__fp16'\00" }
// CHECK-UBSAN: @[[LINE_1100:.*]] = {{.*}}, i32 1100, i32 8 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[LINE_1200:.*]] = {{.*}}, i32 1200, i32 10 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[LINE_1300:.*]] = {{.*}}, i32 1300, i32 10 {{.*}} @{{.*}} }
// CHECK-UBSAN: @[[LINE_1400:.*]] = {{.*}}, i32 1400, i32 10 {{.*}} @{{.*}} }
// Make sure we check the fp16 type_mismatch data so we can easily match the signed char float_cast_overflow
// CHECK-UBSAN: @[[LINE_1500:.*]] = {{.*}}, i32 1500, i32 10 {{.*}} @[[FP16]], {{.*}} }
// CHECK-UBSAN: @[[SCHAR:.*]] = private unnamed_addr constant { i16, i16, [14 x i8] } { i16 0, i16 7, [14 x i8] c"'signed char'\00" }
// CHECK-UBSAN: @[[LINE_1500:.*]] = {{.*}}, i32 1500, i32 10 {{.*}} @[[FP16]], {{.*}} }
// CHECK-UBSAN: @[[LINE_1600:.*]] = {{.*}}, i32 1600, i32 10 {{.*}} @{{.*}} }

// PR6805
// CHECK-COMMON-LABEL: @foo
void foo() {
  union { int i; } u;

  // CHECK-COMMON:      %[[I8PTR:.*]] = bitcast i32* %[[PTR:.*]] to i8*
  // CHECK-COMMON-NEXT: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(i8* %[[I8PTR]], i1 false, i1 false, i1 false)
  // CHECK-COMMON-NEXT: %[[OK:.*]] = icmp uge i64 %[[SIZE]], 4

  // CHECK-UBSAN: br i1 %[[OK]], {{.*}} !prof ![[WEIGHT_MD:.*]], !nosanitize
  // CHECK-TRAP:  br i1 %[[OK]], {{.*}}

  // CHECK-UBSAN:      %[[ARG:.*]] = ptrtoint {{.*}} %[[PTR]] to i64
  // CHECK-UBSAN-NEXT: call void @__ubsan_handle_type_mismatch_v1(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %[[ARG]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW:#[0-9]+]]
  // CHECK-TRAP-NEXT: unreachable
#line 100
  u.i=1;
}

// CHECK-COMMON-LABEL: @bar
int bar(int *a) {
  // CHECK-COMMON:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-COMMON-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK-COMMON:      %[[PTRINT:.*]] = ptrtoint
  // CHECK-COMMON-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-COMMON-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1(i8* bitcast ({{.*}} @[[LINE_200]] to i8*), i64 %[[PTRINT]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable

#line 200
  return *a;
}

// CHECK-UBSAN-LABEL: @addr_space
int addr_space(int __attribute__((address_space(256))) *a) {
  // CHECK-UBSAN-NOT: __ubsan
  return *a;
}

// CHECK-COMMON-LABEL: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK-COMMON:      %[[RHS_INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-COMMON-NEXT: br i1 %[[RHS_INBOUNDS]], label %[[CHECK_BB:.*]], label %[[CONT_BB:.*]],

  // CHECK-COMMON:      [[CHECK_BB]]:
  // CHECK-COMMON-NEXT: %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-COMMON-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-COMMON-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-COMMON-NEXT: br label %[[CONT_BB]]

  // CHECK-COMMON:      [[CONT_BB]]:
  // CHECK-COMMON-NEXT: %[[VALID_BASE:.*]] = phi i1 [ true, {{.*}} ], [ %[[NO_OVERFLOW]], %[[CHECK_BB]] ]
  // CHECK-COMMON-NEXT: %[[VALID:.*]] = and i1 %[[RHS_INBOUNDS]], %[[VALID_BASE]]

  // CHECK-UBSAN: br i1 %[[VALID]], {{.*}} !prof ![[WEIGHT_MD]]
  // CHECK-TRAP:  br i1 %[[VALID]]

  // CHECK-UBSAN:      %[[ARG1:.*]] = zext
  // CHECK-UBSAN-NEXT: %[[ARG2:.*]] = zext
  // CHECK-UBSAN-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])
  // CHECK-UBSAN-NOT:  call void @__ubsan_handle_shift_out_of_bounds

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP:      unreachable
  // CHECK-TRAP-NOT:  call void @llvm.trap()

  // CHECK-COMMON:      %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-COMMON-NEXT: ret i32 %[[RET]]
#line 300
  return a << b;
}

// CHECK-COMMON-LABEL: @rsh_inbounds
int rsh_inbounds(int a, int b) {
  // CHECK-COMMON:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-COMMON:      br i1 %[[INBOUNDS]]

  // CHECK-UBSAN:      %[[ARG1:.*]] = zext
  // CHECK-UBSAN-NEXT: %[[ARG2:.*]] = zext
  // CHECK-UBSAN-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_400]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable

  // CHECK-COMMON:      %[[RET:.*]] = ashr i32 {{.*}}, %[[RHS]]
  // CHECK-COMMON-NEXT: ret i32 %[[RET]]
#line 400
  return a >> b;
}

// CHECK-COMMON-LABEL: @load
int load(int *p) {
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1(i8* bitcast ({{.*}} @[[LINE_500]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 500
  return *p;
}

// CHECK-COMMON-LABEL: @store
void store(int *p, int q) {
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1(i8* bitcast ({{.*}} @[[LINE_600]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 600
  *p = q;
}

struct S { int k; };

// CHECK-COMMON-LABEL: @member_access
int *member_access(struct S *p) {
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1(i8* bitcast ({{.*}} @[[LINE_700]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 700
  return &p->k;
}

// CHECK-COMMON-LABEL: @signed_overflow
int signed_overflow(int a, int b) {
  // CHECK-UBSAN:      %[[ARG1:.*]] = zext
  // CHECK-UBSAN-NEXT: %[[ARG2:.*]] = zext
  // CHECK-UBSAN-NEXT: call void @__ubsan_handle_add_overflow(i8* bitcast ({{.*}} @[[LINE_800]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 800
  return a + b;
}

// CHECK-COMMON-LABEL: @no_return
int no_return() {
  // Reaching the end of a noreturn function is fine in C.
  // FIXME: If the user explicitly requests -fsanitize=return, we should catch
  //        that here even though it's not undefined behavior.
  // CHECK-COMMON-NOT: call
  // CHECK-COMMON-NOT: unreachable
  // CHECK-COMMON: ret i32
}

// CHECK-UBSAN-LABEL: @vla_bound
void vla_bound(int n) {
  // CHECK-UBSAN:      icmp sgt i32 %[[PARAM:.*]], 0
  //
  // CHECK-UBSAN:      %[[ARG:.*]] = zext i32 %[[PARAM]] to i64
  // CHECK-UBSAN-NEXT: call void @__ubsan_handle_vla_bound_not_positive(i8* bitcast ({{.*}} @[[LINE_900]] to i8*), i64 %[[ARG]])
#line 900
  int arr[n * 3];
}

// CHECK-UBSAN-LABEL: @int_float_no_overflow
float int_float_no_overflow(__int128 n) {
  // CHECK-UBSAN-NOT: call void @__ubsan_handle
  return n;
}

// CHECK-COMMON-LABEL: @int_float_overflow
float int_float_overflow(unsigned __int128 n) {
  // This is 2**104. FLT_MAX is 2**128 - 2**104.
  // CHECK-COMMON: %[[INBOUNDS:.*]] = icmp ule i128 %{{.*}}, -20282409603651670423947251286016
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1000]] to i8*),

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1000
  return n;
}

// CHECK-COMMON-LABEL: @int_fp16_overflow
void int_fp16_overflow(int n, __fp16 *p) {
  // CHECK-COMMON: %[[GE:.*]] = icmp sge i32 %{{.*}}, -65504
  // CHECK-COMMON: %[[LE:.*]] = icmp sle i32 %{{.*}}, 65504
  // CHECK-COMMON: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1100]] to i8*),

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1100
  *p = n;
}

// CHECK-COMMON-LABEL: @float_int_overflow
int float_int_overflow(float f) {
  // CHECK-COMMON: %[[GE:.*]] = fcmp ogt float %[[F:.*]], 0xC1E0000020000000
  // CHECK-COMMON: %[[LE:.*]] = fcmp olt float %[[F]], 0x41E0000000000000
  // CHECK-COMMON: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: %[[CAST:.*]] = bitcast float %[[F]] to i32
  // CHECK-UBSAN: %[[ARG:.*]] = zext i32 %[[CAST]] to i64
  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1200]] to i8*), i64 %[[ARG]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1200
  return f;
}

// CHECK-COMMON-LABEL: @long_double_int_overflow
int long_double_int_overflow(long double ld) {
  // CHECK-UBSAN: alloca x86_fp80

  // CHECK-COMMON: %[[GE:.*]] = fcmp ogt x86_fp80 %[[F:.*]], 0xKC01E800000010000000
  // CHECK-COMMON: %[[LE:.*]] = fcmp olt x86_fp80 %[[F]], 0xK401E800000000000000
  // CHECK-COMMON: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: store x86_fp80 %[[F]], x86_fp80* %[[ALLOCA:.*]], align 16, !nosanitize
  // CHECK-UBSAN: %[[ARG:.*]] = ptrtoint x86_fp80* %[[ALLOCA]] to i64
  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1300]] to i8*), i64 %[[ARG]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1300
  return ld;
}

// CHECK-COMMON-LABEL: @float_uint_overflow
unsigned float_uint_overflow(float f) {
  // CHECK-COMMON: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.{{0*}}e+00
  // CHECK-COMMON: %[[LE:.*]] = fcmp olt float %[[F]], 0x41F0000000000000
  // CHECK-COMMON: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1400]] to i8*),

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1400
  return f;
}

// CHECK-COMMON-LABEL: @fp16_char_overflow
signed char fp16_char_overflow(__fp16 *p) {
  // CHECK-COMMON: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.29{{0*}}e+02
  // CHECK-COMMON: %[[LE:.*]] = fcmp olt float %[[F]], 1.28{{0*}}e+02
  // CHECK-COMMON: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1500]] to i8*),

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1500
  return *p;
}

// CHECK-COMMON-LABEL: @float_float_overflow
float float_float_overflow(double f) {
  // CHECK-COMMON: %[[F:.*]] = call double @llvm.fabs.f64(
  // CHECK-COMMON: %[[GE:.*]] = fcmp ogt double %[[F]], 0x47EFFFFFE0000000
  // CHECK-COMMON: %[[LE:.*]] = fcmp olt double %[[F]], 0x7FF0000000000000
  // CHECK-COMMON: %[[OUTOFBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-COMMON: %[[INBOUNDS:.*]] = xor i1 %[[OUTOFBOUNDS]], true
  // CHECK-COMMON-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-UBSAN: call void @__ubsan_handle_float_cast_overflow(i8* bitcast ({{.*}} @[[LINE_1600]] to i8*),

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 1600
  return f;
}

// CHECK-COMMON-LABEL:   @int_divide_overflow
// CHECK-OVERFLOW-LABEL: @int_divide_overflow
int int_divide_overflow(int a, int b) {
  // CHECK-COMMON:         %[[ZERO:.*]] = icmp ne i32 %[[B:.*]], 0
  // CHECK-OVERFLOW-NOT:  icmp ne i32 %{{.*}}, 0

  // CHECK-COMMON:               %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-COMMON-NEXT:          %[[BOK:.*]] = icmp ne i32 %[[B]], -1
  // CHECK-COMMON-NEXT:          %[[OVER:.*]] = or i1 %[[AOK]], %[[BOK]]
  // CHECK-COMMON:         %[[OK:.*]] = and i1 %[[ZERO]], %[[OVER]]
  // CHECK-COMMON:         br i1 %[[OK]]

  // CHECK-OVERFLOW:      %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-OVERFLOW-NEXT: %[[BOK:.*]] = icmp ne i32 %[[B:.*]], -1
  // CHECK-OVERFLOW-NEXT: %[[OK:.*]] = or i1 %[[AOK]], %[[BOK]]
  // CHECK-OVERFLOW:      br i1 %[[OK]]

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return a / b;

  // CHECK-COMMON:          }
  // CHECK-OVERFLOW: }
}

// CHECK-COMMON-LABEL: @sour_bool
_Bool sour_bool(_Bool *p) {
  // CHECK-COMMON: %[[OK:.*]] = icmp ule i8 {{.*}}, 1
  // CHECK-COMMON: br i1 %[[OK]]

  // CHECK-UBSAN: call void @__ubsan_handle_load_invalid_value(i8* bitcast ({{.*}}), i64 {{.*}})

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return *p;
}

// CHECK-COMMON-LABEL: @ret_nonnull
__attribute__((returns_nonnull))
int *ret_nonnull(int *a) {
  // CHECK-COMMON: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK-COMMON: br i1 [[OK]]

  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_return

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return a;
}

// CHECK-COMMON-LABEL: @call_decl_nonnull
__attribute__((nonnull)) void decl_nonnull(int *a);
void call_decl_nonnull(int *a) {
  // CHECK-COMMON: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK-COMMON: br i1 [[OK]]

  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  decl_nonnull(a);
}

extern void *memcpy (void *, const void *, unsigned) __attribute__((nonnull(1, 2)));

// CHECK-COMMON-LABEL: @call_memcpy_nonnull
void call_memcpy_nonnull(void *p, void *q, int sz) {
  // CHECK-COMMON: icmp ne i8* {{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg
  // CHECK-TRAP: call void @llvm.trap()

  // CHECK-COMMON: icmp ne i8* {{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg
  // CHECK-TRAP: call void @llvm.trap()
  memcpy(p, q, sz);
}

extern void *memmove (void *, const void *, unsigned) __attribute__((nonnull(1, 2)));

// CHECK-COMMON-LABEL: @call_memmove_nonnull
void call_memmove_nonnull(void *p, void *q, int sz) {
  // CHECK-COMMON: icmp ne i8* {{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg
  // CHECK-TRAP: call void @llvm.trap()

  // CHECK-COMMON: icmp ne i8* {{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg
  // CHECK-TRAP: call void @llvm.trap()
  memmove(p, q, sz);
}

// CHECK-COMMON-LABEL: @call_nonnull_variadic
__attribute__((nonnull)) void nonnull_variadic(int a, ...);
void call_nonnull_variadic(int a, int *b) {
  // CHECK-COMMON: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK-COMMON: br i1 [[OK]]

  // CHECK-UBSAN: call void @__ubsan_handle_nonnull_arg
  // CHECK-UBSAN-NOT: __ubsan_handle_nonnull_arg

  // CHECK-COMMON: call void (i32, ...) @nonnull_variadic
  nonnull_variadic(a, b);
}

// CHECK-UBSAN: ![[WEIGHT_MD]] = !{!"branch_weights", i32 1048575, i32 1}

// CHECK-TRAP: attributes [[NR_NUW]] = { noreturn nounwind }
