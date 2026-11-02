// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK128
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK128 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK128
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK128 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK256
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK256 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK256
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK256 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK512
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK512 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK512
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK512 --implicit-check-not="not yet implemented"

// A Pure Scalable Type is passed expanded into one scalable vector argument
// per SVE vector or predicate member. A returned Pure Scalable Type with more
// than one member is a packed struct of those vectors. The type is passed
// indirectly if it is unnamed or if it does not fit in the remaining Z (8) or
// P (4) argument registers.
//
// The register sequence is the same at every vector length, so those checks
// are shared. The in-memory layout the sequence is coerced from is not, so a
// check that names it is per-length.

#define N __ARM_FEATURE_SVE_BITS

typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVFloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

typedef struct { fixed_int32_t x; } One;
typedef struct { fixed_int32_t a; fixed_float64_t b; } Two;
typedef struct { fixed_bool_t p; fixed_int32_t v; } PredAndData;
typedef struct { fixed_bool_t a; fixed_bool_t b; } TwoPred;
typedef struct { fixed_bool_t p; } OnePred;
typedef struct { fixed_int32_t x[2]; } Arr2;
typedef struct { Two inner; } Nested;
typedef struct { fixed_int32_t x[9]; } Nine;
typedef struct { fixed_int32_t x; union {} u; } WithEmpty;
typedef struct { fixed_int32_t x; int zla[0]; } WithZero;

// CHECK-LABEL: define{{.*}} void @arg_one(<vscale x 4 x i32> %{{.*}})
void arg_one(One s) {}

// CHECK-LABEL: define{{.*}} <vscale x 4 x i32> @ret_one(<vscale x 4 x i32> %{{.*}})
One ret_one(One s) { return s; }

// CHECK-LABEL: define{{.*}} void @arg_two(<vscale x 4 x i32> %{{.*}}, <vscale x 2 x double> %{{.*}})
void arg_two(Two s) {}

// CHECK-LABEL: define{{.*}} <{ <vscale x 4 x i32>, <vscale x 2 x double> }> @ret_two(<vscale x 4 x i32> %{{.*}}, <vscale x 2 x double> %{{.*}})
Two ret_two(Two s) { return s; }

// Fixed-length predicates are passed as <vscale x 16 x i1>.

// CHECK-LABEL: define{{.*}} void @arg_pred(<vscale x 16 x i1> %{{.*}}, <vscale x 4 x i32> %{{.*}})
void arg_pred(PredAndData s) {}

// CHECK-LABEL: define{{.*}} <{ <vscale x 16 x i1>, <vscale x 4 x i32> }> @ret_pred(<vscale x 16 x i1> %{{.*}}, <vscale x 4 x i32> %{{.*}})
PredAndData ret_pred(PredAndData s) { return s; }

// Arrays and nested records are flattened.

// CHECK-LABEL: define{{.*}} void @arg_arr(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})
void arg_arr(Arr2 a) {}

// CHECK-LABEL: define{{.*}} <{ <vscale x 4 x i32>, <vscale x 4 x i32> }> @ret_arr(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})
Arr2 ret_arr(Arr2 a) { return a; }

// CHECK-LABEL: define{{.*}} void @arg_nested(<vscale x 4 x i32> %{{.*}}, <vscale x 2 x double> %{{.*}})
void arg_nested(Nested s) {}

// A zero-length array is empty and does not disqualify a Pure Scalable Type.

// CHECK-LABEL: define{{.*}} void @arg_zero(<vscale x 4 x i32> %{{.*}})
void arg_zero(WithZero s) {}

// CHECK-LABEL: define{{.*}} <vscale x 4 x i32> @ret_zero(<vscale x 4 x i32> %{{.*}})
WithZero ret_zero(WithZero s) { return s; }

// An empty C union member is skipped.

// CHECK-LABEL: define{{.*}} void @arg_empty(<vscale x 4 x i32> %{{.*}})
void arg_empty(WithEmpty s) {}

// CHECK-LABEL: define{{.*}} <vscale x 4 x i32> @ret_empty(<vscale x 4 x i32> %{{.*}})
WithEmpty ret_empty(WithEmpty s) { return s; }

// A type with more than 8 data vectors never fits in registers.

// CHECK128-LABEL: define{{.*}} void @arg_nine(ptr {{.*}}align 16 dead_on_return dereferenceable(144) %{{.*}})
// CHECK256-LABEL: define{{.*}} void @arg_nine(ptr {{.*}}align 16 dead_on_return dereferenceable(288) %{{.*}})
// CHECK512-LABEL: define{{.*}} void @arg_nine(ptr {{.*}}align 16 dead_on_return dereferenceable(576) %{{.*}})
void arg_nine(Nine s) {}

// CHECK-LABEL: define{{.*}} void @ret_nine(ptr {{.*}}sret(%struct.Nine) align 16 %{{.*}})
Nine ret_nine(void) { Nine s = {}; return s; }

// A Pure Scalable Type that does not fit in the remaining registers is passed
// indirectly and does not consume registers, so a later one can still fit.

// CHECK128-LABEL: define{{.*}} void @budget(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c, <vscale x 4 x i32> %d, <vscale x 4 x i32> %e, <vscale x 4 x i32> %f, <vscale x 4 x i32> %g, ptr {{.*}}align 16 dead_on_return dereferenceable(32) %two, <vscale x 4 x i32> %{{.*}})
// CHECK256-LABEL: define{{.*}} void @budget(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c, <vscale x 4 x i32> %d, <vscale x 4 x i32> %e, <vscale x 4 x i32> %f, <vscale x 4 x i32> %g, ptr {{.*}}align 16 dead_on_return dereferenceable(64) %two, <vscale x 4 x i32> %{{.*}})
// CHECK512-LABEL: define{{.*}} void @budget(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c, <vscale x 4 x i32> %d, <vscale x 4 x i32> %e, <vscale x 4 x i32> %f, <vscale x 4 x i32> %g, ptr {{.*}}align 16 dead_on_return dereferenceable(128) %two, <vscale x 4 x i32> %{{.*}})
void budget(__SVInt32_t a, __SVInt32_t b, __SVInt32_t c, __SVInt32_t d,
            __SVInt32_t e, __SVInt32_t f, __SVInt32_t g, Two two, One one) {}

// CHECK128-LABEL: define{{.*}} void @pred_budget(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b, <vscale x 16 x i1> %c, ptr {{.*}}align 2 dead_on_return dereferenceable(4) %both, <vscale x 16 x i1> %{{.*}})
// CHECK256-LABEL: define{{.*}} void @pred_budget(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b, <vscale x 16 x i1> %c, ptr {{.*}}align 2 dead_on_return dereferenceable(8) %both, <vscale x 16 x i1> %{{.*}})
// CHECK512-LABEL: define{{.*}} void @pred_budget(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b, <vscale x 16 x i1> %c, ptr {{.*}}align 2 dead_on_return dereferenceable(16) %both, <vscale x 16 x i1> %{{.*}})
void pred_budget(__SVBool_t a, __SVBool_t b, __SVBool_t c,
                 TwoPred both, OnePred one) {}

// An unnamed Pure Scalable Type is passed indirectly.

void variadic_callee(One named, ...);

// CHECK-LABEL: define{{.*}} void @test_variadic(
// CHECK128: call void (<vscale x 4 x i32>, ...) @variadic_callee(<vscale x 4 x i32> %{{.*}}, ptr {{.*}}align 16 dead_on_return dereferenceable(16) %{{.*}})
// CHECK256: call void (<vscale x 4 x i32>, ...) @variadic_callee(<vscale x 4 x i32> %{{.*}}, ptr {{.*}}align 16 dead_on_return dereferenceable(32) %{{.*}})
// CHECK512: call void (<vscale x 4 x i32>, ...) @variadic_callee(<vscale x 4 x i32> %{{.*}}, ptr {{.*}}align 16 dead_on_return dereferenceable(64) %{{.*}})
void test_variadic(One *a, One *b) { variadic_callee(*a, *b); }

// An SVE tuple is passed directly if it fits in the remaining registers and
// indirectly otherwise.

// CHECK-LABEL: define{{.*}} void @arg_tuple(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})
void arg_tuple(__clang_svint32x2_t v) {}

// CHECK-LABEL: define{{.*}} { <vscale x 4 x i32>, <vscale x 4 x i32> } @ret_tuple(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})
__clang_svint32x2_t ret_tuple(__clang_svint32x2_t v) { return v; }

// CHECK-LABEL: define{{.*}} void @tuple_budget(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c, <vscale x 4 x i32> %d, <vscale x 4 x i32> %e, <vscale x 4 x i32> %f, <vscale x 4 x i32> %g, ptr {{.*}}align 16 dead_on_return %{{.*}})
void tuple_budget(__SVInt32_t a, __SVInt32_t b, __SVInt32_t c, __SVInt32_t d,
                  __SVInt32_t e, __SVInt32_t f, __SVInt32_t g,
                  __clang_svint32x2_t t) {}

// CHECK-LABEL: define{{.*}} void @pred_tuple(<vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, ptr {{.*}}align 2 dead_on_return %{{.*}})
void pred_tuple(__clang_svboolx4_t a, __clang_svboolx2_t b) {}

// A fixed-length SVE type is 16-byte aligned however wide it is, but the
// vector it lowers to is aligned to its size rounded up to a power of two. A
// record holding one wider than 16 bytes is therefore packed once lowered, and
// every gap in a packed record is an explicit array.

// A packed record is lowered with explicit padding for every gap, including
// one that falls at an offset the source alignment already reaches. The
// coerce type carries those arrays. The return type is the unpadded register
// sequence.

typedef struct __attribute__((packed)) {
  fixed_bool_t p;
  fixed_int32_t misaligned;
  fixed_int32_t natural __attribute__((aligned(16)));
  fixed_bool_t tail;
} PackedGap;

// CHECK-LABEL: define{{.*}} <{ <vscale x 16 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 16 x i1> }> @ret_packed_gap()
// CHECK128: getelementptr inbounds nuw { <2 x i8>, <4 x i32>, [14 x i8], <4 x i32>, <2 x i8>, [14 x i8] }, ptr %retval, i32 0, i32 0
// CHECK256: getelementptr inbounds nuw { <4 x i8>, <8 x i32>, [12 x i8], <8 x i32>, <4 x i8>, [12 x i8] }, ptr %retval, i32 0, i32 0
// CHECK512: getelementptr inbounds nuw { <8 x i8>, <16 x i32>, [8 x i8], <16 x i32>, <8 x i8>, [8 x i8] }, ptr %retval, i32 0, i32 0
PackedGap ret_packed_gap(void) { PackedGap s = {}; return s; }

// CHECK-LABEL: define{{.*}} void @arg_packed_gap(<vscale x 16 x i1> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 16 x i1> %{{.*}})
// CHECK128: getelementptr inbounds nuw { <2 x i8>, <4 x i32>, [14 x i8], <4 x i32>, <2 x i8>, [14 x i8] }, ptr %s, i32 0, i32 0
// CHECK256: getelementptr inbounds nuw { <4 x i8>, <8 x i32>, [12 x i8], <8 x i32>, <4 x i8>, [12 x i8] }, ptr %s, i32 0, i32 0
// CHECK512: getelementptr inbounds nuw { <8 x i8>, <16 x i32>, [8 x i8], <16 x i32>, <8 x i8>, [8 x i8] }, ptr %s, i32 0, i32 0
void arg_packed_gap(PackedGap s) {}

// A trailing predicate leaves the record with tail padding. A 16-byte data
// vector rounds the members up to the record size and the coerce type has no
// array for the tail. A wider one packs the record, and then the tail is an
// explicit array.

typedef struct { fixed_int32_t v; fixed_bool_t p; } DataThenPred;

// CHECK-LABEL: define{{.*}} <{ <vscale x 4 x i32>, <vscale x 16 x i1> }> @ret_data_then_pred(<vscale x 4 x i32> %{{.*}}, <vscale x 16 x i1> %{{.*}})
// CHECK128: getelementptr inbounds nuw { <4 x i32>, <2 x i8> }, ptr %s, i32 0, i32 0
// CHECK256: getelementptr inbounds nuw { <8 x i32>, <4 x i8>, [12 x i8] }, ptr %s, i32 0, i32 0
// CHECK512: getelementptr inbounds nuw { <16 x i32>, <8 x i8>, [8 x i8] }, ptr %s, i32 0, i32 0
DataThenPred ret_data_then_pred(DataThenPred s) { return s; }

// An over-aligned record is not packed. Its tail padding is an explicit array
// in the coerce type where the member alignment does not account for it, and
// a data vector at least as wide as the record alignment leaves no tail at
// all.

typedef struct __attribute__((aligned(32))) { fixed_int32_t x; } OverAligned;

// CHECK-LABEL: define{{.*}} <vscale x 4 x i32> @ret_overaligned(<vscale x 4 x i32> %{{.*}})
// CHECK128: getelementptr inbounds nuw { <4 x i32>, [16 x i8] }, ptr %s, i32 0, i32 0
// CHECK256: getelementptr inbounds nuw { <8 x i32> }, ptr %s, i32 0, i32 0
// CHECK512: getelementptr inbounds nuw { <16 x i32> }, ptr %s, i32 0, i32 0
OverAligned ret_overaligned(OverAligned s) { return s; }

typedef struct __attribute__((aligned(64))) {
  fixed_int32_t v;
  fixed_bool_t p;
} OverAlignedTail;

// CHECK-LABEL: define{{.*}} <{ <vscale x 4 x i32>, <vscale x 16 x i1> }> @ret_overaligned_tail(<vscale x 4 x i32> %{{.*}}, <vscale x 16 x i1> %{{.*}})
// CHECK128: getelementptr inbounds nuw { <4 x i32>, <2 x i8>, [46 x i8] }, ptr %s, i32 0, i32 0
// CHECK256: getelementptr inbounds nuw { <8 x i32>, <4 x i8>, [28 x i8] }, ptr %s, i32 0, i32 0
// CHECK512: getelementptr inbounds nuw { <16 x i32>, <8 x i8>, [56 x i8] }, ptr %s, i32 0, i32 0
OverAlignedTail ret_overaligned_tail(OverAlignedTail s) { return s; }
