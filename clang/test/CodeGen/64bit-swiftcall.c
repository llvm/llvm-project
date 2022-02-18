// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin10 -target-cpu core2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin10 -target-cpu core2 -emit-llvm -o - %s | FileCheck %s --check-prefix=X86-64
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple arm64-apple-ios9 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple arm64-apple-ios9 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM64

// REQUIRES: aarch64-registered-target,x86-registered-target

#define SWIFTCALL __attribute__((swiftcall))
#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))
#define ASYNC_CONTEXT __attribute__((swift_async_context))

// CHECK-DAG: %struct.atomic_padded = type { { %struct.packed, [7 x i8] } }
// CHECK-DAG: %struct.packed = type <{ i64, i8 }>
//
// CHECK: [[STRUCT2_RESULT:@.*]] = private {{.*}} constant [[STRUCT2_TYPE:%.*]] { i32 0, i8 0, i8 undef, i8 0, i32 0, i32 0 }

/*****************************************************************************/
/****************************** PARAMETER ABIS *******************************/
/*****************************************************************************/

SWIFTCALL void indirect_result_1(OUT int *arg0, OUT float *arg1) {}
// CHECK-LABEL: define {{.*}} void @indirect_result_1(i32* noalias sret(i32*) align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

// TODO: maybe this shouldn't suppress sret.
SWIFTCALL int indirect_result_2(OUT int *arg0, OUT float *arg1) {  __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} i32 @indirect_result_2(i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

typedef struct { char array[1024]; } struct_reallybig;
SWIFTCALL struct_reallybig indirect_result_3(OUT int *arg0, OUT float *arg1) { __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} void @indirect_result_3({{.*}}* noalias sret(%struct.struct_reallybig) {{.*}}, i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

SWIFTCALL void context_1(CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_1(i8* swiftself

SWIFTASYNCCALL void async_context_1(ASYNC_CONTEXT void *ctx) {}
// CHECK-LABEL: define {{.*}} void @async_context_1(i8* swiftasync

SWIFTCALL void context_2(void *arg0, CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_2(i8*{{.*}}, i8* swiftself

SWIFTASYNCCALL void async_context_2(void *arg0, ASYNC_CONTEXT void *ctx) {}
// CHECK-LABEL: define {{.*}} void @async_context_2(i8*{{.*}}, i8* swiftasync

SWIFTCALL void context_error_1(CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_1(i32* swiftself{{.*}}, float** swifterror %0)
// CHECK:       [[TEMP:%.*]] = alloca float*, align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[ERRORARG:%.*]], align 8
// CHECK:       store float* [[T0]], float** [[TEMP]], align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 8
// CHECK:       store float* [[T0]], float** [[ERRORARG]], align 8
void test_context_error_1() {
  int x;
  float *error;
  context_error_1(&x, &error);
}
// CHECK-LABEL: define{{.*}} void @test_context_error_1()
// CHECK:       [[X:%.*]] = alloca i32, align 4
// CHECK:       [[ERROR:%.*]] = alloca float*, align 8
// CHECK:       [[TEMP:%.*]] = alloca swifterror float*, align 8
// CHECK:       [[T0:%.*]] = load float*, float** [[ERROR]], align 8
// CHECK:       store float* [[T0]], float** [[TEMP]], align 8
// CHECK:       call [[SWIFTCC:swiftcc]] void @context_error_1(i32* swiftself [[X]], float** swifterror [[TEMP]])
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 8
// CHECK:       store float* [[T0]], float** [[ERROR]], align 8

SWIFTCALL void context_error_2(short s, CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_2(i16{{.*}}, i32* swiftself{{.*}}, float** swifterror %0)

/*****************************************************************************/
/********************************** LOWERING *********************************/
/*****************************************************************************/

typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int5 __attribute__((ext_vector_type(5)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef char char16 __attribute__((ext_vector_type(16)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef long long long2 __attribute__((ext_vector_type(2)));

#define TEST(TYPE)                       \
  SWIFTCALL TYPE return_##TYPE(void) {   \
    TYPE result = {};                    \
    return result;                       \
  }                                      \
  SWIFTCALL void take_##TYPE(TYPE v) {   \
  }                                      \
  void test_##TYPE() {                   \
    take_##TYPE(return_##TYPE());        \
  }

/*****************************************************************************/
/*********************************** STRUCTS *********************************/
/*****************************************************************************/

typedef struct {
} struct_empty;
TEST(struct_empty);
// CHECK-LABEL: define {{.*}} @return_struct_empty()
// CHECK:   ret void
// CHECK-LABEL: define {{.*}} @take_struct_empty()
// CHECK:   ret void

typedef struct {
  int x;
  char c0;
  char c1;
  int f0;
  int f1;
} struct_1;
TEST(struct_1);
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64 } @return_struct_1() {{.*}}{
// CHECK:   [[RET:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   call void @llvm.memset
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* %retval to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = load i64, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = load i64, i64* [[GEP1]], align 4
// CHECK:   [[R0:%.*]] = insertvalue { i64, i64 } undef, i64 [[T0]], 0
// CHECK:   [[R1:%.*]] = insertvalue { i64, i64 } [[R0]], i64 [[T1]], 1
// CHECK:   ret { i64, i64 } [[R1]]
// CHECK: }
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_1(i64 %0, i64 %1) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* [[V]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   store i64 %1, i64* [[GEP1]], align 4
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define{{.*}} void @test_struct_1() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[STRUCT1:%.*]], align 4
// CHECK:   [[RET:%.*]] = call swiftcc { i64, i64 } @return_struct_1()
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT1]]* [[AGG]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[E0:%.*]] = extractvalue { i64, i64 } [[RET]], 0
// CHECK:   store i64 [[E0]], i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[E1:%.*]] = extractvalue { i64, i64 } [[RET]], 1
// CHECK:   store i64 [[E1]], i64* [[GEP1]], align 4
// CHECK:   [[CAST2:%.*]] = bitcast [[STRUCT1]]* [[AGG]] to { i64, i64 }*
// CHECK:   [[GEP2:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST2]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[GEP2]], align 4
// CHECK:   [[GEP3:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST2]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load i64, i64* [[GEP3]], align 4
// CHECK:   call swiftcc void @take_struct_1(i64 [[V0]], i64 [[V1]])
// CHECK:   ret void
// CHECK: }

typedef struct {
  int x;
  char c0;
  __attribute__((aligned(2))) char c1;
  int f0;
  int f1;
} struct_2;
TEST(struct_2);
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64 } @return_struct_2() {{.*}}{
// CHECK:   [[RET:%.*]] = alloca [[STRUCT2_TYPE]], align 4
// CHECK:   [[CASTVAR:%.*]] = bitcast {{.*}} [[RET]]
// CHECK:   call void @llvm.memcpy{{.*}}({{.*}}[[CASTVAR]], {{.*}}[[STRUCT2_RESULT]]
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT2_TYPE]]* [[RET]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = load i64, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = load i64, i64* [[GEP1]], align 4
// CHECK:   [[R0:%.*]] = insertvalue { i64, i64 } undef, i64 [[T0]], 0
// CHECK:   [[R1:%.*]] = insertvalue { i64, i64 } [[R0]], i64 [[T1]], 1
// CHECK:   ret { i64, i64 } [[R1]]
// CHECK: }
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_2(i64 %0, i64 %1) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT:%.*]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT]]* [[V]] to { i64, i64 }*
// CHECK:   [[GEP0:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP0]], align 4
// CHECK:   [[GEP1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   store i64 %1, i64* [[GEP1]], align 4
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define{{.*}} void @test_struct_2() {{.*}} {
// CHECK:   [[TMP:%.*]] = alloca [[STRUCT2_TYPE]], align 4
// CHECK:   [[CALL:%.*]] = call swiftcc { i64, i64 } @return_struct_2()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[STRUCT2_TYPE]]* [[TMP]] to { i64, i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds {{.*}} [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T0:%.*]] = extractvalue { i64, i64 } [[CALL]], 0
// CHECK:   store i64 [[T0]], i64* [[GEP]], align 4
// CHECK:   [[GEP:%.*]] = getelementptr inbounds {{.*}} [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T0:%.*]] = extractvalue { i64, i64 } [[CALL]], 1
// CHECK:   store i64 [[T0]], i64* [[GEP]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT2_TYPE]]* [[TMP]] to { i64, i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 0
// CHECK:   [[R0:%.*]] = load i64, i64* [[GEP]], align 4
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[CAST]], i32 0, i32 1
// CHECK:   [[R1:%.*]] = load i64, i64* [[GEP]], align 4
// CHECK:   call swiftcc void @take_struct_2(i64 [[R0]], i64 [[R1]])
// CHECK:   ret void
// CHECK: }

// There's no way to put a field randomly in the middle of an otherwise
// empty storage unit in C, so that case has to be tested in C++, which
// can use empty structs to introduce arbitrary padding.  (In C, they end up
// with size 0 and so don't affect layout.)

// Misaligned data rule.
typedef struct {
  char c0;
  __attribute__((packed)) float f;
} struct_misaligned_1;
TEST(struct_misaligned_1)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_misaligned_1()
// CHECK:  [[RET:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:  [[CAST:%.*]] = bitcast [[STRUCT]]* [[RET]] to i8*
// CHECK:  call void @llvm.memset{{.*}}(i8* align 1 [[CAST]], i8 0, i64 5
// CHECK:  [[CAST:%.*]] = bitcast [[STRUCT]]* [[RET]] to { i64 }*
// CHECK:  [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:  [[R0:%.*]] = load i64, i64* [[GEP]], align 1
// CHECK:  ret i64 [[R0]]
// CHECK:}
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_misaligned_1(i64 %0) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:   [[CAST:%.*]] = bitcast [[STRUCT]]* [[V]] to { i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP]], align 1
// CHECK:   ret void
// CHECK: }
// CHECK: define{{.*}} void @test_struct_misaligned_1() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[STRUCT:%.*]], align 1
// CHECK:   [[CALL:%.*]] = call swiftcc i64 @return_struct_misaligned_1()
// CHECK:   [[T0:%.*]] = bitcast [[STRUCT]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   store i64 [[CALL]], i64* [[T1]], align 1
// CHECK:   [[T0:%.*]] = bitcast [[STRUCT]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   [[P:%.*]] = load i64, i64* [[T1]], align 1
// CHECK:   call swiftcc void @take_struct_misaligned_1(i64 [[P]])
// CHECK:   ret void
// CHECK: }

// Too many scalars.
typedef struct {
  long long x[5];
} struct_big_1;
TEST(struct_big_1)

// CHECK-LABEL: define {{.*}} void @return_struct_big_1({{.*}} noalias sret

// Should not be byval.
// CHECK-LABEL: define {{.*}} void @take_struct_big_1({{.*}}*{{( %.*)?}})

/*****************************************************************************/
/********************************* TYPE MERGING ******************************/
/*****************************************************************************/

typedef union {
  float f;
  double d;
} union_het_fp;
TEST(union_het_fp)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_union_het_fp()
// CHECK:  [[RET:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:  [[CAST:%.*]] = bitcast [[UNION]]* [[RET]] to i8*
// CHECK:  call void @llvm.memcpy{{.*}}(i8* align 8 [[CAST]]
// CHECK:  [[CAST:%.*]] = bitcast [[UNION]]* [[RET]] to { i64 }*
// CHECK:  [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:  [[R0:%.*]] = load i64, i64* [[GEP]], align 8
// CHECK:  ret i64 [[R0]]
// CHECK-LABEL: define{{.*}} swiftcc void @take_union_het_fp(i64 %0) {{.*}}{
// CHECK:   [[V:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[V]] to { i64 }*
// CHECK:   [[GEP:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[CAST]], i32 0, i32 0
// CHECK:   store i64 %0, i64* [[GEP]], align 8
// CHECK:   ret void
// CHECK: }
// CHECK-LABEL: define{{.*}} void @test_union_het_fp() {{.*}}{
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 8
// CHECK:   [[CALL:%.*]] = call swiftcc i64 @return_union_het_fp()
// CHECK:   [[T0:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   store i64 [[CALL]], i64* [[T1]], align 8
// CHECK:   [[T0:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64 }*
// CHECK:   [[T1:%.*]] = getelementptr inbounds { i64 }, { i64 }* [[T0]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[T1]], align 8
// CHECK:   call swiftcc void @take_union_het_fp(i64 [[V0]])
// CHECK:   ret void
// CHECK: }


typedef union {
  float f1;
  float f2;
} union_hom_fp;
TEST(union_hom_fp)
// CHECK-LABEL: define{{.*}} void @test_union_hom_fp()
// CHECK:   [[TMP:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] float @return_union_hom_fp()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG:{ float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store float [[CALL]], float* [[T0]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_union_hom_fp(float [[FIRST]])
// CHECK:   ret void

typedef union {
  float f1;
  float4 fv2;
} union_hom_fp_partial;
TEST(union_hom_fp_partial)
// CHECK: define{{.*}} void @test_union_hom_fp_partial()
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call swiftcc { float, float, float, float } @return_union_hom_fp_partial()
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { float, float, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 0
// CHECK:   store float [[T1]], float* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 1
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue { float, float, float, float } [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { float, float, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load float, float* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load float, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[V2:%.*]] = load float, float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* [[CAST]], i32 0, i32 3
// CHECK:   [[V3:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call swiftcc void @take_union_hom_fp_partial(float [[V0]], float [[V1]], float [[V2]], float [[V3]])
// CHECK:   ret void
// CHECK: }

typedef union {
  struct { int x, y; } f1;
  float4 fv2;
} union_het_fpv_partial;
TEST(union_het_fpv_partial)
// CHECK-LABEL: define{{.*}} void @test_union_het_fpv_partial()
// CHECK:   [[AGG:%.*]] = alloca [[UNION:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call swiftcc { i64, float, float } @return_union_het_fpv_partial()
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 0
// CHECK:   store i64 [[T1]], i64* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 1
// CHECK:   store float [[T1]], float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue { i64, float, float } [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST:%.*]] = bitcast [[UNION]]* [[AGG]] to { i64, float, float }*
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 0
// CHECK:   [[V0:%.*]] = load i64, i64* [[T0]], align 16
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 1
// CHECK:   [[V1:%.*]] = load float, float* [[T0]], align 8
// CHECK:   [[T0:%.*]] = getelementptr inbounds { i64, float, float }, { i64, float, float }* [[CAST]], i32 0, i32 2
// CHECK:   [[V2:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call swiftcc void @take_union_het_fpv_partial(i64 [[V0]], float [[V1]], float [[V2]])
// CHECK:   ret void
// CHECK: }

/*****************************************************************************/
/****************************** VECTOR LEGALIZATION **************************/
/*****************************************************************************/

TEST(int4)
// CHECK-LABEL: define {{.*}} <4 x i32> @return_int4()
// CHECK-LABEL: define {{.*}} @take_int4(<4 x i32>

TEST(int8)
// CHECK-LABEL: define {{.*}} @return_int8()
// CHECK:   [[RET:%.*]] = alloca [[REC:<8 x i32>]], align 16
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align
// CHECK:   store
// CHECK:   load
// CHECK:   store
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ <4 x i32>, <4 x i32> }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ <4 x i32>, <4 x i32> }]] undef, <4 x i32> [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], <4 x i32> [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_int8(<4 x i32> %0, <4 x i32> %1)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store <4 x i32> %1, <4 x i32>* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define{{.*}} void @test_int8()
// CHECK:   [[TMP1:%.*]] = alloca [[REC]], align
// CHECK:   [[TMP2:%.*]] = alloca [[REC]], align
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_int8()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP1]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[V:%.*]] = load [[REC]], [[REC]]* [[TMP1]], align
// CHECK:   store [[REC]] [[V]], [[REC]]* [[TMP2]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP2]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_int8(<4 x i32> [[FIRST]], <4 x i32> [[SECOND]])
// CHECK:   ret void

TEST(int5)
// CHECK-LABEL: define {{.*}} @return_int5()
// CHECK:   [[RET:%.*]] = alloca [[REC:<5 x i32>]], align 16
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align
// CHECK:   store
// CHECK:   load
// CHECK:   store
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ <4 x i32>, i32 }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ <4 x i32>, i32 }]] undef, <4 x i32> [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i32 [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_int5(<4 x i32> %0, i32 %1)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i32 %1, i32* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define{{.*}} void @test_int5()
// CHECK:   [[TMP1:%.*]] = alloca [[REC]], align
// CHECK:   [[TMP2:%.*]] = alloca [[REC]], align
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_int5()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP1]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store <4 x i32> [[T1]], <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i32 [[T1]], i32* [[T0]], align
// CHECK:   [[V:%.*]] = load [[REC]], [[REC]]* [[TMP1]], align
// CHECK:   store [[REC]] [[V]], [[REC]]* [[TMP2]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP2]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load <4 x i32>, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_int5(<4 x i32> [[FIRST]], i32 [[SECOND]])
// CHECK:   ret void

typedef struct {
  int x;
  int3 v __attribute__((packed));
} misaligned_int3;
TEST(misaligned_int3)
// CHECK-LABEL: define{{.*}} swiftcc void @take_misaligned_int3(i64 %0, i64 %1)

typedef struct {
  float f0;
} struct_f1;
TEST(struct_f1)
// CHECK-LABEL: define{{.*}} swiftcc float @return_struct_f1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_f1(float %0)

typedef struct {
  float f0;
  float f1;
} struct_f2;
TEST(struct_f2)
// CHECK-LABEL: define{{.*}} swiftcc { float, float } @return_struct_f2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_f2(float %0, float %1)

typedef struct {
  float f0;
  float f1;
  float f2;
} struct_f3;
TEST(struct_f3)
// CHECK-LABEL: define{{.*}} swiftcc { float, float, float } @return_struct_f3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_f3(float %0, float %1, float %2)

typedef struct {
  float f0;
  float f1;
  float f2;
  float f3;
} struct_f4;
TEST(struct_f4)
// CHECK-LABEL: define{{.*}} swiftcc { float, float, float, float } @return_struct_f4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_f4(float %0, float %1, float %2, float %3)


typedef struct {
  double d0;
} struct_d1;
TEST(struct_d1)
// CHECK-LABEL: define{{.*}} swiftcc double @return_struct_d1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_d1(double %0)

typedef struct {
  double d0;
  double d1;
} struct_d2;
TEST(struct_d2)

// CHECK-LABEL: define{{.*}} swiftcc { double, double } @return_struct_d2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_d2(double %0, double %1)
typedef struct {
  double d0;
  double d1;
  double d2;
} struct_d3;
TEST(struct_d3)
// CHECK-LABEL: define{{.*}} swiftcc { double, double, double } @return_struct_d3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_d3(double %0, double %1, double %2)

typedef struct {
  double d0;
  double d1;
  double d2;
  double d3;
} struct_d4;
TEST(struct_d4)
// CHECK-LABEL: define{{.*}} swiftcc { double, double, double, double } @return_struct_d4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_d4(double %0, double %1, double %2, double %3)

typedef struct {
  double d0;
  double d1;
  double d2;
  double d3;
  double d4;
} struct_d5;
TEST(struct_d5)
// CHECK: define{{.*}} swiftcc void @return_struct_d5([[STRUCT5:%.*]]* noalias sret([[STRUCT5]])
// CHECK: define{{.*}} swiftcc void @take_struct_d5([[STRUCT5]]

typedef struct {
  char c0;
} struct_c1;
TEST(struct_c1)
// CHECK-LABEL: define{{.*}} swiftcc i8 @return_struct_c1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c1(i8 %0)

typedef struct {
  char c0;
  char c1;
} struct_c2;
TEST(struct_c2)
// CHECK-LABEL: define{{.*}} swiftcc i16 @return_struct_c2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c2(i16 %0)
//

typedef struct {
  char c0;
  char c1;
  char c2;
} struct_c3;
TEST(struct_c3)
// CHECK-LABEL: define{{.*}} swiftcc i32 @return_struct_c3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c3(i32 %0)

typedef struct {
  char c0;
  char c1;
  char c2;
  char c3;
} struct_c4;
TEST(struct_c4)
// CHECK-LABEL: define{{.*}} swiftcc i32 @return_struct_c4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c4(i32 %0)

typedef struct {
  char c0;
  char c1;
  char c2;
  char c3;
  char c4;
} struct_c5;
TEST(struct_c5)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_c5()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c5(i64 %0)
//
typedef struct {
  char c0;
  char c1;
  char c2;
  char c3;
  char c4;
  char c5;
  char c6;
  char c7;
  char c8;
} struct_c9;
TEST(struct_c9)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i8 } @return_struct_c9()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_c9(i64 %0, i8 %1)

typedef struct {
  short s0;
} struct_s1;
TEST(struct_s1)
// CHECK-LABEL: define{{.*}} swiftcc i16 @return_struct_s1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_s1(i16 %0)

typedef struct {
  short s0;
  short s1;
} struct_s2;
TEST(struct_s2)
// CHECK-LABEL: define{{.*}} swiftcc i32 @return_struct_s2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_s2(i32 %0)
//

typedef struct {
  short s0;
  short s1;
  short s2;
} struct_s3;
TEST(struct_s3)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_s3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_s3(i64 %0)

typedef struct {
  short s0;
  short s1;
  short s2;
  short s3;
} struct_s4;
TEST(struct_s4)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_s4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_s4(i64 %0)

typedef struct {
  short s0;
  short s1;
  short s2;
  short s3;
  short s4;
} struct_s5;
TEST(struct_s5)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i16 } @return_struct_s5()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_s5(i64 %0, i16 %1)


typedef struct {
  int i0;
} struct_i1;
TEST(struct_i1)
// CHECK-LABEL: define{{.*}} swiftcc i32 @return_struct_i1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_i1(i32 %0)

typedef struct {
  int i0;
  int i1;
} struct_i2;
TEST(struct_i2)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_i2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_i2(i64 %0)

typedef struct {
  int i0;
  int i1;
  int i2;
} struct_i3;
TEST(struct_i3)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i32 } @return_struct_i3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_i3(i64 %0, i32 %1)

typedef struct {
  int i0;
  int i1;
  int i2;
  int i3;
} struct_i4;
TEST(struct_i4)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64 } @return_struct_i4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_i4(i64 %0, i64 %1)

typedef struct {
  long long l0;
} struct_l1;
TEST(struct_l1)
// CHECK-LABEL: define{{.*}} swiftcc i64 @return_struct_l1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_l1(i64 %0)

typedef struct {
  long long l0;
  long long l1;
} struct_l2;
TEST(struct_l2)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64 } @return_struct_l2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_l2(i64 %0, i64 %1)

typedef struct {
  long long l0;
  long long l1;
  long long l2;
} struct_l3;
TEST(struct_l3)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64, i64 } @return_struct_l3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_l3(i64 %0, i64 %1, i64 %2)

typedef struct {
  long long l0;
  long long l1;
  long long l2;
  long long l3;
} struct_l4;
TEST(struct_l4)
// CHECK-LABEL: define{{.*}} swiftcc { i64, i64, i64, i64 } @return_struct_l4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_l4(i64 %0, i64 %1, i64 %2, i64 %3)

typedef struct {
  long long l0;
  long long l1;
  long long l2;
  long long l3;
  long long l4;
} struct_l5;
TEST(struct_l5)
// CHECK: define{{.*}} swiftcc void @return_struct_l5([[STRUCT5:%.*]]* noalias sret([[STRUCT5]])
// CHECK: define{{.*}} swiftcc void @take_struct_l5([[STRUCT5]]*

typedef struct {
  char16 c0;
} struct_vc1;
TEST(struct_vc1)
// CHECK-LABEL: define{{.*}} swiftcc <16 x i8> @return_struct_vc1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vc1(<16 x i8> %0)

typedef struct {
  char16 c0;
  char16 c1;
} struct_vc2;
TEST(struct_vc2)
// CHECK-LABEL: define{{.*}} swiftcc { <16 x i8>, <16 x i8> } @return_struct_vc2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vc2(<16 x i8> %0, <16 x i8> %1)

typedef struct {
  char16 c0;
  char16 c1;
  char16 c2;
} struct_vc3;
TEST(struct_vc3)
// CHECK-LABEL: define{{.*}} swiftcc { <16 x i8>, <16 x i8>, <16 x i8> } @return_struct_vc3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vc3(<16 x i8> %0, <16 x i8> %1, <16 x i8> %2)

typedef struct {
  char16 c0;
  char16 c1;
  char16 c2;
  char16 c3;
} struct_vc4;
TEST(struct_vc4)
// CHECK-LABEL: define{{.*}} swiftcc { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @return_struct_vc4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vc4(<16 x i8> %0, <16 x i8> %1, <16 x i8> %2, <16 x i8> %3)

typedef struct {
  char16 c0;
  char16 c1;
  char16 c2;
  char16 c3;
  char16 c4;
} struct_vc5;
TEST(struct_vc5)
// CHECK: define{{.*}} swiftcc void @return_struct_vc5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vc5([[STRUCT]]

typedef struct {
  short8 c0;
} struct_vs1;
TEST(struct_vs1)
// CHECK-LABEL: define{{.*}} swiftcc <8 x i16> @return_struct_vs1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vs1(<8 x i16> %0)

typedef struct {
  short8 c0;
  short8 c1;
} struct_vs2;
TEST(struct_vs2)
// CHECK-LABEL: define{{.*}} swiftcc { <8 x i16>, <8 x i16> } @return_struct_vs2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vs2(<8 x i16> %0, <8 x i16> %1)

typedef struct {
  short8 c0;
  short8 c1;
  short8 c2;
} struct_vs3;
TEST(struct_vs3)
// CHECK-LABEL: define{{.*}} swiftcc { <8 x i16>, <8 x i16>, <8 x i16> } @return_struct_vs3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vs3(<8 x i16> %0, <8 x i16> %1, <8 x i16> %2)

typedef struct {
  short8 c0;
  short8 c1;
  short8 c2;
  short8 c3;
} struct_vs4;
TEST(struct_vs4)
// CHECK-LABEL: define{{.*}} swiftcc { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @return_struct_vs4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vs4(<8 x i16> %0, <8 x i16> %1, <8 x i16> %2, <8 x i16> %3)

typedef struct {
  short8 c0;
  short8 c1;
  short8 c2;
  short8 c3;
  short8 c4;
} struct_vs5;
TEST(struct_vs5)
// CHECK: define{{.*}} swiftcc void @return_struct_vs5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vs5([[STRUCT]]

typedef struct {
  int4 c0;
} struct_vi1;
TEST(struct_vi1)
// CHECK-LABEL: define{{.*}} swiftcc <4 x i32> @return_struct_vi1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vi1(<4 x i32> %0)

typedef struct {
  int4 c0;
  int4 c1;
} struct_vi2;
TEST(struct_vi2)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x i32>, <4 x i32> } @return_struct_vi2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vi2(<4 x i32> %0, <4 x i32> %1)

typedef struct {
  int4 c0;
  int4 c1;
  int4 c2;
} struct_vi3;
TEST(struct_vi3)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x i32>, <4 x i32>, <4 x i32> } @return_struct_vi3()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vi3(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2)

typedef struct {
  int4 c0;
  int4 c1;
  int4 c2;
  int4 c3;
} struct_vi4;
TEST(struct_vi4)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @return_struct_vi4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vi4(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2, <4 x i32> %3)

typedef struct {
  int4 c0;
  int4 c1;
  int4 c2;
  int4 c3;
  int4 c4;
} struct_vi5;
TEST(struct_vi5)
// CHECK: define{{.*}} swiftcc void @return_struct_vi5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vi5([[STRUCT]]

typedef struct {
  long2 c0;
} struct_vl1;
TEST(struct_vl1)
// CHECK-LABEL: define{{.*}} swiftcc <2 x i64> @return_struct_vl1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vl1(<2 x i64> %0)

typedef struct {
  long2 c0;
  long2 c1;
  long2 c2;
  long2 c3;
} struct_vl4;
TEST(struct_vl4)
// CHECK-LABEL: define{{.*}} swiftcc { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @return_struct_vl4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vl4(<2 x i64> %0, <2 x i64> %1, <2 x i64> %2, <2 x i64> %3)

typedef struct {
  long2 c0;
  long2 c1;
  long2 c2;
  long2 c3;
  long2 c4;
} struct_vl5;
TEST(struct_vl5)
// CHECK: define{{.*}} swiftcc void @return_struct_vl5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vl5([[STRUCT]]

typedef struct {
  double2 c0;
} struct_vd1;
TEST(struct_vd1)
// CHECK-LABEL: define{{.*}} swiftcc <2 x double> @return_struct_vd1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vd1(<2 x double> %0)

typedef struct {
  double2 c0;
  double2 c1;
  double2 c2;
  double2 c3;
} struct_vd4;
TEST(struct_vd4)
// CHECK-LABEL: define{{.*}} swiftcc { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @return_struct_vd4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vd4(<2 x double> %0, <2 x double> %1, <2 x double> %2, <2 x double> %3)

typedef struct {
  double2 c0;
  double2 c1;
  double2 c2;
  double2 c3;
  double2 c4;
} struct_vd5;
TEST(struct_vd5)
// CHECK: define{{.*}} swiftcc void @return_struct_vd5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vd5([[STRUCT]]

typedef struct {
  double4 c0;
} struct_vd41;
TEST(struct_vd41)
// CHECK-LABEL: define{{.*}} swiftcc { <2 x double>, <2 x double> } @return_struct_vd41()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vd41(<2 x double> %0, <2 x double> %1)

typedef struct {
  double4 c0;
  double4 c1;
} struct_vd42;
TEST(struct_vd42)
// CHECK-LABEL: define{{.*}} swiftcc { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @return_struct_vd42()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vd42(<2 x double> %0, <2 x double> %1, <2 x double> %2, <2 x double> %3)

typedef struct {
  double4 c0;
  double4 c1;
  double4 c2;
} struct_vd43;
TEST(struct_vd43)
// CHECK: define{{.*}} swiftcc void @return_struct_vd43([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vd43([[STRUCT]]

typedef struct {
  float4 c0;
} struct_vf1;
TEST(struct_vf1)
// CHECK-LABEL: define{{.*}} swiftcc <4 x float> @return_struct_vf1()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vf1(<4 x float> %0)

typedef struct {
  float4 c0;
  float4 c1;
} struct_vf2;
TEST(struct_vf2)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x float>, <4 x float> } @return_struct_vf2()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vf2(<4 x float> %0, <4 x float> %1)

typedef struct {
  float4 c0;
  float4 c1;
  float4 c2;
  float4 c3;
} struct_vf4;
TEST(struct_vf4)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @return_struct_vf4()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vf4(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x float> %3)

typedef struct {
  float4 c0;
  float4 c1;
  float4 c2;
  float4 c3;
  float4 c4;
} struct_vf5;
TEST(struct_vf5)
// CHECK: define{{.*}} swiftcc void @return_struct_vf5([[STRUCT:%.*]]* noalias sret([[STRUCT]])
// CHECK: define{{.*}} swiftcc void @take_struct_vf5([[STRUCT]]

typedef struct {
  float8 c0;
} struct_vf81;
TEST(struct_vf81)
// CHECK-LABEL: define{{.*}} swiftcc { <4 x float>, <4 x float> } @return_struct_vf81()
// CHECK-LABEL: define{{.*}} swiftcc void @take_struct_vf81(<4 x float> %0, <4 x float> %1)

// Don't crash.
typedef union {
int4 v[2];
struct {
  int LSW;
  int d7;
  int d6;
  int d5;
  int d4;
  int d3;
  int d2;
  int MSW;
} s;
} union_het_vecint;
TEST(union_het_vecint)
// CHECK: define{{.*}} swiftcc void @return_union_het_vecint([[UNION:%.*]]* noalias sret([[UNION]])
// CHECK: define{{.*}} swiftcc void @take_union_het_vecint([[UNION]]*

typedef struct {
  float3 f3;
} struct_v1f3;
TEST(struct_v1f3)
// ARM64-LABEL: define{{.*}} swiftcc { <2 x float>, float } @return_struct_v1f3()
// ARM64-LABEL: define{{.*}} swiftcc void @take_struct_v1f3(<2 x float> %0, float %1)

typedef struct {
  int3 vect;
  unsigned long long val;
} __attribute__((packed)) padded_alloc_size_vector;
TEST(padded_alloc_size_vector)
// X86-64-LABEL: take_padded_alloc_size_vector(<3 x i32> %0, i64 %1)
// X86-64-NOT: [4 x i8]
// x86-64: ret void

typedef union {
  float f1;
  float3 fv2;
} union_hom_fp_partial2;
TEST(union_hom_fp_partial2)
// X86-64-LABEL: take_union_hom_fp_partial2(float %0, float %1, float %2)
// ARM64-LABEL: take_union_hom_fp_partial2(float %0, float %1, float %2)

// At one point, we emitted lifetime.ends without a matching lifetime.start for
// CoerceAndExpanded args. Since we're not performing optimizations, neither
// intrinsic should be emitted.
// CHECK-LABEL: define{{.*}} void @no_lifetime_markers
void no_lifetime_markers() {
  // CHECK-NOT: call void @llvm.lifetime.
  take_int5(return_int5());
}

typedef struct {
  unsigned long long a;
  unsigned long long b;
} double_word;

typedef struct {
  _Atomic(double_word) a;
} atomic_double_word;

// CHECK-LABEL: use_atomic(i64 %0, i64 %1)
SWIFTCALL void use_atomic(atomic_double_word a) {}

typedef struct {
  unsigned long long a;
  unsigned char b;
} __attribute__((packed)) packed;

typedef struct {
  _Atomic(packed) a;
} atomic_padded;

// CHECK-LABEL: use_atomic_padded(i64 %0, i64 %1)
SWIFTCALL void use_atomic_padded(atomic_padded a) {}
