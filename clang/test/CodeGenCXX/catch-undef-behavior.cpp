// RUN: %clang_cc1 -std=c++11 -fsanitize=signed-integer-overflow,integer-divide-by-zero,float-divide-by-zero,shift-base,shift-exponent,unreachable,return,vla-bound,alignment,null,vptr,object-size,float-cast-overflow,bool,enum,array-bounds,function -fsanitize-recover=signed-integer-overflow,integer-divide-by-zero,float-divide-by-zero,shift-base,shift-exponent,vla-bound,alignment,null,vptr,object-size,float-cast-overflow,bool,enum,array-bounds,function -emit-llvm %s -o - -triple x86_64-linux-gnu | opt -instnamer -S | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -fsanitize=vptr,address -fsanitize-recover=vptr,address -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang_cc1 -std=c++11 -fsanitize=vptr -fsanitize-recover=vptr -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=DOWNCAST-NULL
// RUN: %clang_cc1 -std=c++11 -fsanitize=function -emit-llvm %s -o - -triple x86_64-linux-gnux32 | FileCheck %s --check-prefix=CHECK-X32
// RUN: %clang_cc1 -std=c++11 -fsanitize=function -emit-llvm %s -o - -triple i386-linux-gnu | FileCheck %s --check-prefix=CHECK-X86

struct S {
  double d;
  int a, b;
  virtual int f();
};

// Check that type descriptor global is not modified by ASan.
// CHECK-ASAN: [[TYPE_DESCR:@[0-9]+]] = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'S'\00" }

// Check that type mismatch handler is not modified by ASan.
// CHECK-ASAN: private unnamed_addr global { { [{{.*}} x i8]*, i32, i32 }, { i16, i16, [4 x i8] }*, i8*, i8 } { {{.*}}, { i16, i16, [4 x i8] }* [[TYPE_DESCR]], {{.*}} }

struct T : S {};

// CHECK-LABEL: @_Z17reference_binding
void reference_binding(int *p, S *q) {
  // C++ core issue 453: If an lvalue to which a reference is directly bound
  // designates neither an existing object or function of an appropriate type,
  // nor a region of storage of suitable size and alignment to contain an object
  // of the reference's type, the behavior is undefined.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  int &r = *p;

  // A reference is not required to refer to an object within its lifetime.
  // CHECK-NOT: __ubsan_handle_dynamic_type_cache_miss
  S &r2 = *q;
}

// CHECK-LABEL: @_Z13member_access
// CHECK-ASAN-LABEL: @_Z13member_access
void member_access(S *p) {
  // (1a) Check 'p' is appropriately sized and aligned for member access.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 24

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // (1b) Check that 'p' actually points to an 'S'.

  // CHECK: %[[VPTRADDR:.*]] = bitcast {{.*}} to i64*
  // CHECK-NEXT: %[[VPTR:.*]] = load i64, i64* %[[VPTRADDR]]
  //
  // hash_16_bytes:
  //
  // If this number changes, it indicates that either the mangled name of ::S
  // has changed, or that LLVM's hashing function has changed. The latter case
  // is OK if the hashing function is still stable.
  //
  // The two hash values are for 64- and 32-bit Clang binaries, respectively.
  // FIXME: We should produce a 64-bit value either way.
  //
  // CHECK-NEXT: xor i64 {{-4030275160588942838|2562089159}}, %[[VPTR]]
  // CHECK-NEXT: mul i64 {{.*}}, -7070675565921424023
  // CHECK-NEXT: lshr i64 {{.*}}, 47
  // CHECK-NEXT: xor i64
  // CHECK-NEXT: xor i64 %[[VPTR]]
  // CHECK-NEXT: mul i64 {{.*}}, -7070675565921424023
  // CHECK-NEXT: lshr i64 {{.*}}, 47
  // CHECK-NEXT: xor i64
  // CHECK-NEXT: %[[HASH:.*]] = mul i64 {{.*}}, -7070675565921424023
  //
  // Check the hash against the table:
  //
  // CHECK-NEXT: %[[IDX:.*]] = and i64 %{{.*}}, 127
  // CHECK-NEXT: getelementptr inbounds [128 x i64], [128 x i64]* @__ubsan_vptr_type_cache, i32 0, i64 %[[IDX]]
  // CHECK-NEXT: %[[CACHEVAL:.*]] = load i64, i64*
  // CHECK-NEXT: icmp eq i64 %[[CACHEVAL]], %[[HASH]]
  // CHECK-NEXT: br i1

  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss({{.*}}, i64 %{{.*}}, i64 %[[HASH]])
  // CHECK-NOT: unreachable
  // CHECK: {{.*}}:

  // (2) Check 'p->b' is appropriately sized and aligned for a load.

  // FIXME: Suppress this in the trivial case of a member access, because we
  // know we've just checked the member access expression itself.

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  int k = p->b;

  // (3a) Check 'p' is appropriately sized and aligned for member function call.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 24

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // (3b) Check that 'p' actually points to an 'S'

  // CHECK: load i64, i64*
  // CHECK-NEXT: xor i64 {{-4030275160588942838|2562089159}},
  // [...]
  // CHECK: getelementptr inbounds [128 x i64], [128 x i64]* @__ubsan_vptr_type_cache, i32 0, i64 %
  // CHECK: br i1
  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss({{.*}}, i64 %{{.*}}, i64 %{{.*}})
  // CHECK-NOT: unreachable
  // CHECK: {{.*}}:

  k = p->f();
}

// CHECK-LABEL: @_Z12lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK: %[[RHS_INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[RHS_INBOUNDS]], label %[[CHECK_BB:.*]], label %[[CONT_BB:.*]],

  // CHECK:      [[CHECK_BB]]:
  // CHECK-NEXT: %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]

  // This is present for C++11 but not for C: C++ core issue 1457 allows a '1'
  // to be shifted into the sign bit, but not out of it.
  // CHECK-NEXT: %[[SHIFTED_OUT_NOT_SIGN:.*]] = lshr i32 %[[SHIFTED_OUT]], 1

  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT_NOT_SIGN]], 0
  // CHECK-NEXT: br label %[[CONT_BB]]

  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: %[[VALID_BASE:.*]] = phi i1 [ true, {{.*}} ], [ %[[NO_OVERFLOW]], %[[CHECK_BB]] ]
  // CHECK-NEXT: %[[VALID:.*]] = and i1 %[[RHS_INBOUNDS]], %[[VALID_BASE]]
  // CHECK-NEXT: br i1 %[[VALID]]

  // CHECK: call void @__ubsan_handle_shift_out_of_bounds
  // CHECK-NOT: call void @__ubsan_handle_shift_out_of_bounds

  // CHECK: %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
  return a << b;
}

// CHECK-LABEL: @_Z9no_return
int no_return() {
  // CHECK:      call void @__ubsan_handle_missing_return(i8* bitcast ({{.*}}* @{{.*}} to i8*)) [[NR_NUW:#[0-9]+]]
  // CHECK-NEXT: unreachable
}

// CHECK-LABEL: @_Z9sour_bool
bool sour_bool(bool *p) {
  // CHECK: %[[OK:.*]] = icmp ule i8 {{.*}}, 1
  // CHECK: br i1 %[[OK]]
  // CHECK: call void @__ubsan_handle_load_invalid_value(i8* bitcast ({{.*}}), i64 {{.*}})
  return *p;
}

enum E1 { e1a = 0, e1b = 127 } e1;
enum E2 { e2a = -1, e2b = 64 } e2;
enum E3 { e3a = (1u << 31) - 1 } e3;

// CHECK-LABEL: @_Z14bad_enum_value
int bad_enum_value() {
  // CHECK: %[[E1:.*]] = icmp ule i32 {{.*}}, 127
  // CHECK: br i1 %[[E1]]
  // CHECK: call void @__ubsan_handle_load_invalid_value(
  int a = e1;

  // CHECK: %[[E2HI:.*]] = icmp sle i32 {{.*}}, 127
  // CHECK: %[[E2LO:.*]] = icmp sge i32 {{.*}}, -128
  // CHECK: %[[E2:.*]] = and i1 %[[E2HI]], %[[E2LO]]
  // CHECK: br i1 %[[E2]]
  // CHECK: call void @__ubsan_handle_load_invalid_value(
  int b = e2;

  // CHECK: %[[E3:.*]] = icmp ule i32 {{.*}}, 2147483647
  // CHECK: br i1 %[[E3]]
  // CHECK: call void @__ubsan_handle_load_invalid_value(
  int c = e3;
  return a + b + c;
}

// CHECK-LABEL: @_Z20bad_downcast_pointer
// DOWNCAST-NULL-LABEL: @_Z20bad_downcast_pointer
void bad_downcast_pointer(S *p) {
  // CHECK: %[[NONNULL:.*]] = icmp ne {{.*}}, null
  // CHECK: br i1 %[[NONNULL]],

  // A null poiner access is guarded without -fsanitize=null.
  // DOWNCAST-NULL: %[[NONNULL:.*]] = icmp ne {{.*}}, null
  // DOWNCAST-NULL: br i1 %[[NONNULL]],

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(
  // CHECK: %[[E1:.*]] = icmp uge i64 %[[SIZE]], 24
  // CHECK: %[[MISALIGN:.*]] = and i64 %{{.*}}, 7
  // CHECK: %[[E2:.*]] = icmp eq i64 %[[MISALIGN]], 0
  // CHECK: %[[E12:.*]] = and i1 %[[E1]], %[[E2]]
  // CHECK: br i1 %[[E12]],

  // CHECK: call void @__ubsan_handle_type_mismatch
  // CHECK: br label

  // CHECK: br i1 %{{.*}},

  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss
  // CHECK: br label
  (void) static_cast<T*>(p);
}

// CHECK-LABEL: @_Z22bad_downcast_reference
void bad_downcast_reference(S &p) {
  // CHECK: %[[E1:.*]] = icmp ne {{.*}}, null
  // CHECK-NOT: br i1

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(
  // CHECK: %[[E2:.*]] = icmp uge i64 %[[SIZE]], 24

  // CHECK: %[[MISALIGN:.*]] = and i64 %{{.*}}, 7
  // CHECK: %[[E3:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // CHECK: %[[E12:.*]] = and i1 %[[E1]], %[[E2]]
  // CHECK: %[[E123:.*]] = and i1 %[[E12]], %[[E3]]
  // CHECK: br i1 %[[E123]],

  // CHECK: call void @__ubsan_handle_type_mismatch
  // CHECK: br label

  // CHECK: br i1 %{{.*}},

  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss
  // CHECK: br label
  (void) static_cast<T&>(p);
}

// CHECK-LABEL: @_Z11array_index
int array_index(const int (&a)[4], int n) {
  // CHECK: %[[K1_OK:.*]] = icmp ult i64 %{{.*}}, 4
  // CHECK: br i1 %[[K1_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  int k1 = a[n];

  // CHECK: %[[R1_OK:.*]] = icmp ule i64 %{{.*}}, 4
  // CHECK: br i1 %[[R1_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  const int *r1 = &a[n];

  // CHECK: %[[K2_OK:.*]] = icmp ult i64 %{{.*}}, 8
  // CHECK: br i1 %[[K2_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  int k2 = ((const int(&)[8])a)[n];

  // CHECK: %[[K3_OK:.*]] = icmp ult i64 %{{.*}}, 4
  // CHECK: br i1 %[[K3_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  int k3 = n[a];

  return k1 + *r1 + k2;
}

// CHECK-LABEL: @_Z17multi_array_index
int multi_array_index(int n, int m) {
  int arr[4][6];

  // CHECK: %[[IDX1_OK:.*]] = icmp ult i64 %{{.*}}, 4
  // CHECK: br i1 %[[IDX1_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(

  // CHECK: %[[IDX2_OK:.*]] = icmp ult i64 %{{.*}}, 6
  // CHECK: br i1 %[[IDX2_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  return arr[n][m];
}

// CHECK-LABEL: @_Z11array_arith
int array_arith(const int (&a)[4], int n) {
  // CHECK: %[[K1_OK:.*]] = icmp ule i64 %{{.*}}, 4
  // CHECK: br i1 %[[K1_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  const int *k1 = a + n;

  // CHECK: %[[K2_OK:.*]] = icmp ule i64 %{{.*}}, 8
  // CHECK: br i1 %[[K2_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  const int *k2 = (const int(&)[8])a + n;

  return *k1 + *k2;
}

struct ArrayMembers {
  int a1[5];
  int a2[1];
};
// CHECK-LABEL: @_Z18struct_array_index
int struct_array_index(ArrayMembers *p, int n) {
  // CHECK: %[[IDX_OK:.*]] = icmp ult i64 %{{.*}}, 5
  // CHECK: br i1 %[[IDX_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  return p->a1[n];
}

// CHECK-LABEL: @_Z16flex_array_index
int flex_array_index(ArrayMembers *p, int n) {
  // CHECK-NOT: call void @__ubsan_handle_out_of_bounds(
  return p->a2[n];
}

extern int incomplete[];
// CHECK-LABEL: @_Z22incomplete_array_index
int incomplete_array_index(int n) {
  // CHECK-NOT: call void @__ubsan_handle_out_of_bounds(
  return incomplete[n];
}

typedef __attribute__((ext_vector_type(4))) int V4I;
// CHECK-LABEL: @_Z12vector_index
int vector_index(V4I v, int n) {
  // CHECK: %[[IDX_OK:.*]] = icmp ult i64 %{{.*}}, 4
  // CHECK: br i1 %[[IDX_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  return v[n];
}

// CHECK-LABEL: @_Z12string_index
char string_index(int n) {
  // CHECK: %[[IDX_OK:.*]] = icmp ult i64 %{{.*}}, 6
  // CHECK: br i1 %[[IDX_OK]]
  // CHECK: call void @__ubsan_handle_out_of_bounds(
  return "Hello"[n];
}

class A // align=4
{
  int a1, a2, a3;
};

class B // align=8
{
  long b1, b2;
};

class C : public A, public B // align=16
{
  alignas(16) int c1;
};

// Make sure we check the alignment of the pointer after subtracting any
// offset. The pointer before subtraction doesn't need to be aligned for
// the destination type.

// CHECK-LABEL: define void @_Z16downcast_pointerP1B(%class.B* %b)
void downcast_pointer(B *b) {
  (void) static_cast<C*>(b);
  // Alignment check from EmitTypeCheck(TCK_DowncastPointer, ...)
  // CHECK: [[SUB:%[.a-z0-9]*]] = getelementptr i8, i8* {{.*}}, i64 -16
  // CHECK-NEXT: [[C:%.+]] = bitcast i8* [[SUB]] to %class.C*
  // null check goes here
  // CHECK: [[FROM_PHI:%.+]] = phi %class.C* [ [[C]], {{.*}} ], {{.*}}
  // Objectsize check goes here
  // CHECK: [[C_INT:%.+]] = ptrtoint %class.C* [[FROM_PHI]] to i64
  // CHECK-NEXT: [[MASKED:%.+]] = and i64 [[C_INT]], 15
  // CHECK-NEXT: [[TEST:%.+]] = icmp eq i64 [[MASKED]], 0
  // AND the alignment test with the objectsize test.
  // CHECK-NEXT: [[AND:%.+]] = and i1 {{.*}}, [[TEST]]
  // CHECK-NEXT: br i1 [[AND]]
}

// CHECK-LABEL: define void @_Z18downcast_referenceR1B(%class.B* dereferenceable({{[0-9]+}}) %b)
void downcast_reference(B &b) {
  (void) static_cast<C&>(b);
  // Alignment check from EmitTypeCheck(TCK_DowncastReference, ...)
  // CHECK:      [[SUB:%[.a-z0-9]*]] = getelementptr i8, i8* {{.*}}, i64 -16
  // CHECK-NEXT: [[C:%.+]] = bitcast i8* [[SUB]] to %class.C*
  // Objectsize check goes here
  // CHECK:      [[C_INT:%.+]] = ptrtoint %class.C* [[C]] to i64
  // CHECK-NEXT: [[MASKED:%.+]] = and i64 [[C_INT]], 15
  // CHECK-NEXT: [[TEST:%.+]] = icmp eq i64 [[MASKED]], 0
  // AND the alignment test with the objectsize test.
  // CHECK:      [[AND:%.+]] = and i1 {{.*}}, [[TEST]]
  // CHECK-NEXT: br i1 [[AND]]
}

// CHECK-LABEL: @_Z22indirect_function_callPFviE({{.*}} prologue <{ i32, i8* }> <{ i32 1413876459, i8* bitcast ({ i8*, i8* }* @_ZTIFvPFviEE to i8*) }>
// CHECK-X32: @_Z22indirect_function_callPFviE({{.*}} prologue <{ i32, i8* }> <{ i32 1413875435, i8* bitcast ({ i8*, i8* }* @_ZTIFvPFviEE to i8*) }>
// CHECK-X86: @_Z22indirect_function_callPFviE({{.*}} prologue <{ i32, i8* }> <{ i32 1413875435, i8* bitcast ({ i8*, i8* }* @_ZTIFvPFviEE to i8*) }>
void indirect_function_call(void (*p)(int)) {
  // CHECK: [[PTR:%.+]] = bitcast void (i32)* {{.*}} to <{ i32, i8* }>*

  // Signature check
  // CHECK-NEXT: [[SIGPTR:%.+]] = getelementptr <{ i32, i8* }>, <{ i32, i8* }>* [[PTR]], i32 0, i32 0
  // CHECK-NEXT: [[SIG:%.+]] = load i32, i32* [[SIGPTR]]
  // CHECK-NEXT: [[SIGCMP:%.+]] = icmp eq i32 [[SIG]], 1413876459
  // CHECK-NEXT: br i1 [[SIGCMP]]

  // RTTI pointer check
  // CHECK: [[RTTIPTR:%.+]] = getelementptr <{ i32, i8* }>, <{ i32, i8* }>* [[PTR]], i32 0, i32 1
  // CHECK-NEXT: [[RTTI:%.+]] = load i8*, i8** [[RTTIPTR]]
  // CHECK-NEXT: [[RTTICMP:%.+]] = icmp eq i8* [[RTTI]], bitcast ({ i8*, i8* }* @_ZTIFviE to i8*)
  // CHECK-NEXT: br i1 [[RTTICMP]]
  p(42);
}

namespace UpcastPointerTest {
struct S {};
struct T : S { double d; };
struct V : virtual S {};

// CHECK-LABEL: upcast_pointer
S* upcast_pointer(T* t) {
  // Check for null pointer
  // CHECK: %[[NONNULL:.*]] = icmp ne {{.*}}, null
  // CHECK: br i1 %[[NONNULL]]

  // Check alignment
  // CHECK: %[[MISALIGN:.*]] = and i64 %{{.*}}, 7
  // CHECK: icmp eq i64 %[[MISALIGN]], 0

  // CHECK: call void @__ubsan_handle_type_mismatch
  return t;
}

V getV();

// CHECK-LABEL: upcast_to_vbase
void upcast_to_vbase() {
  // No need to check for null here, as we have a temporary here.

  // CHECK-NOT: br i1

  // CHECK: call i64 @llvm.objectsize
  // CHECK: call void @__ubsan_handle_type_mismatch
  // CHECK: call void @__ubsan_handle_dynamic_type_cache_miss
  const S& s = getV();
}
}

struct ThisAlign {
  void this_align_lambda();
  void this_align_lambda_2();
};
void ThisAlign::this_align_lambda() {
  // CHECK-LABEL: define {{.*}}@"_ZZN9ThisAlign17this_align_lambdaEvENK3$_0clEv"
  // CHECK-SAME: (%{{.*}}* %[[this:[^)]*]])
  // CHECK: %[[this_addr:.*]] = alloca
  // CHECK: store %{{.*}}* %[[this]], %{{.*}}** %[[this_addr]],
  // CHECK: %[[this_inner:.*]] = load %{{.*}}*, %{{.*}}** %[[this_addr]],
  // CHECK: %[[this_outer_addr:.*]] = getelementptr inbounds %{{.*}}, %{{.*}}* %[[this_inner]], i32 0, i32 0
  // CHECK: %[[this_outer:.*]] = load %{{.*}}*, %{{.*}}** %[[this_outer_addr]],
  //
  // CHECK: %[[this_inner_isnonnull:.*]] = icmp ne %{{.*}}* %[[this_inner]], null
  // CHECK: %[[this_inner_asint:.*]] = ptrtoint %{{.*}}* %[[this_inner]] to i
  // CHECK: %[[this_inner_misalignment:.*]] = and i{{32|64}} %[[this_inner_asint]], {{3|7}},
  // CHECK: %[[this_inner_isaligned:.*]] = icmp eq i{{32|64}} %[[this_inner_misalignment]], 0
  // CHECK: %[[this_inner_valid:.*]] = and i1 %[[this_inner_isnonnull]], %[[this_inner_isaligned]],
  // CHECK: br i1 %[[this_inner_valid:.*]]
  [&] { return this; } ();
}

namespace CopyValueRepresentation {
  // CHECK-LABEL: define {{.*}} @_ZN23CopyValueRepresentation2S3aSERKS0_
  // CHECK-NOT: call {{.*}} @__ubsan_handle_load_invalid_value
  // CHECK-LABEL: define {{.*}} @_ZN23CopyValueRepresentation2S4aSEOS0_
  // CHECK-NOT: call {{.*}} @__ubsan_handle_load_invalid_value
  // CHECK-LABEL: define {{.*}} @_ZN23CopyValueRepresentation2S1C2ERKS0_
  // CHECK-NOT: call {{.*}} __ubsan_handle_load_invalid_value
  // CHECK-LABEL: define {{.*}} @_ZN23CopyValueRepresentation2S2C2ERKS0_
  // CHECK: __ubsan_handle_load_invalid_value
  // CHECK-LABEL: define {{.*}} @_ZN23CopyValueRepresentation2S5C2ERKS0_
  // CHECK-NOT: call {{.*}} __ubsan_handle_load_invalid_value

  struct CustomCopy { CustomCopy(); CustomCopy(const CustomCopy&); };
  struct S1 {
    CustomCopy CC;
    bool b;
  };
  void callee1(S1);
  void test1() {
    S1 s11;
    callee1(s11);
    S1 s12;
    s12 = s11;
  }

  static bool some_global_bool;
  struct ExprCopy {
    ExprCopy();
    ExprCopy(const ExprCopy&, bool b = some_global_bool);
  };
  struct S2 {
    ExprCopy EC;
    bool b;
  };
  void callee2(S2);
  void test2(void) {
    S2 s21;
    callee2(s21);
    S2 s22;
    s22 = s21;
  }

  struct CustomAssign { CustomAssign &operator=(const CustomAssign&); };
  struct S3 {
    CustomAssign CA;
    bool b;
  };
  void test3() {
    S3 x, y;
    x = y;
  }

  struct CustomMove {
    CustomMove();
    CustomMove(const CustomMove&&);
    CustomMove &operator=(const CustomMove&&);
  };
  struct S4 {
    CustomMove CM;
    bool b;
  };
  void test4() {
    S4 x, y;
    x = static_cast<S4&&>(y);
  }

  struct EnumCustomCopy {
    EnumCustomCopy();
    EnumCustomCopy(const EnumCustomCopy&);
  };
  struct S5 {
    EnumCustomCopy ECC;
    bool b;
  };
  void callee5(S5);
  void test5() {
    S5 s51;
    callee5(s51);
    S5 s52;
    s52 = s51;
  }
}

void ThisAlign::this_align_lambda_2() {
  // CHECK-LABEL: define {{.*}}@"_ZZN9ThisAlign19this_align_lambda_2EvENK3$_1clEv"
  // CHECK-SAME: (%{{.*}}* %[[this:[^)]*]])
  // CHECK: %[[this_addr:.*]] = alloca
  // CHECK: store %{{.*}}* %[[this]], %{{.*}}** %[[this_addr]],
  // CHECK: %[[this_inner:.*]] = load %{{.*}}*, %{{.*}}** %[[this_addr]],
  //
  // Do not perform a null check on the 'this' pointer if the function might be
  // called from a static invoker.
  // CHECK-NOT: icmp ne %{{.*}}* %[[this_inner]], null
  auto *p = +[] {};
  p();
}

// CHECK: attributes [[NR_NUW]] = { noreturn nounwind }
