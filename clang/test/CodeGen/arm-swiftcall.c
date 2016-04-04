// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// This isn't really testing anything ARM-specific; it's just a convenient
// 32-bit platform.

#define SWIFTCALL __attribute__((swiftcall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))

/*****************************************************************************/
/****************************** PARAMETER ABIS *******************************/
/*****************************************************************************/

SWIFTCALL void indirect_result_1(OUT int *arg0, OUT float *arg1) {}
// CHECK-LABEL: define {{.*}} void @indirect_result_1(i32* noalias sret align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

// TODO: maybe this shouldn't suppress sret.
SWIFTCALL int indirect_result_2(OUT int *arg0, OUT float *arg1) {  __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} i32 @indirect_result_2(i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

typedef struct { char array[1024]; } struct_reallybig;
SWIFTCALL struct_reallybig indirect_result_3(OUT int *arg0, OUT float *arg1) { __builtin_unreachable(); }
// CHECK-LABEL: define {{.*}} void @indirect_result_3({{.*}}* noalias sret {{.*}}, i32* noalias align 4 dereferenceable(4){{.*}}, float* noalias align 4 dereferenceable(4){{.*}})

SWIFTCALL void context_1(CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_1(i8* swiftself

SWIFTCALL void context_2(void *arg0, CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_2(i8*{{.*}}, i8* swiftself

SWIFTCALL void context_error_1(CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_1(i32* swiftself{{.*}}, float** swifterror)
// CHECK:       [[TEMP:%.*]] = alloca float*, align 4
// CHECK:       [[T0:%.*]] = load float*, float** [[ERRORARG:%.*]], align 4
// CHECK:       store float* [[T0]], float** [[TEMP]], align 4
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 4
// CHECK:       store float* [[T0]], float** [[ERRORARG]], align 4
void test_context_error_1() {
  int x;
  float *error;
  context_error_1(&x, &error);
}
// CHECK-LABEL: define void @test_context_error_1()
// CHECK:       [[X:%.*]] = alloca i32, align 4
// CHECK:       [[ERROR:%.*]] = alloca float*, align 4
// CHECK:       [[TEMP:%.*]] = alloca swifterror float*, align 4
// CHECK:       [[T0:%.*]] = load float*, float** [[ERROR]], align 4
// CHECK:       store float* [[T0]], float** [[TEMP]], align 4
// CHECK:       call [[SWIFTCC:swiftcc]] void @context_error_1(i32* swiftself [[X]], float** swifterror [[TEMP]])
// CHECK:       [[T0:%.*]] = load float*, float** [[TEMP]], align 4
// CHECK:       store float* [[T0]], float** [[ERROR]], align 4

SWIFTCALL void context_error_2(short s, CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_2(i16{{.*}}, i32* swiftself{{.*}}, float** swifterror)

/*****************************************************************************/
/********************************** LOWERING *********************************/
/*****************************************************************************/

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int5 __attribute__((ext_vector_type(5)));
typedef int int8 __attribute__((ext_vector_type(8)));

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
  float f0;
  float f1;
} struct_1;
TEST(struct_1);
// CHECK-LABEL: define {{.*}} @return_struct_1()
// CHECK:   [[RET:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align 4
// CHECK:   @llvm.memset
// CHECK:   @llvm.memcpy
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ i32, i16, \[2 x i8\], float, float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i16, i16* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 4
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ i32, i16, float, float }]] undef, i32 [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i16 [[SECOND]], 1
// CHECK:   [[T2:%.*]] = insertvalue [[UAGG]] [[T1]], float [[THIRD]], 2
// CHECK:   [[T3:%.*]] = insertvalue [[UAGG]] [[T2]], float [[FOURTH]], 3
// CHECK:   ret [[UAGG]] [[T3]]
// CHECK-LABEL: define {{.*}} @take_struct_1(i32, i16, float, float)
// CHECK:   [[V:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store i32 %0, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i16 %1, i16* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   store float %2, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 4
// CHECK:   store float %3, float* [[T0]], align 4
// CHECK:   ret void
// CHECK-LABEL: define void @test_struct_1()
// CHECK:   [[TMP:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_struct_1()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store i32 [[T1]], i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i16 [[T1]], i16* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 4
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i16, i16* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 4
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_struct_1(i32 [[FIRST]], i16 [[SECOND]], float [[THIRD]], float [[FOURTH]])
// CHECK:   ret void

typedef struct {
  int x;
  char c0;
  __attribute__((aligned(2))) char c1;
  float f0;
  float f1;
} struct_2;
TEST(struct_2);
// CHECK-LABEL: define {{.*}} @return_struct_2()
// CHECK:   [[RET:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align 4
// CHECK:   @llvm.memcpy
// CHECK:   @llvm.memcpy
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ i32, i32, float, float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ i32, i32, float, float }]] undef, i32 [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i32 [[SECOND]], 1
// CHECK:   [[T2:%.*]] = insertvalue [[UAGG]] [[T1]], float [[THIRD]], 2
// CHECK:   [[T3:%.*]] = insertvalue [[UAGG]] [[T2]], float [[FOURTH]], 3
// CHECK:   ret [[UAGG]] [[T3]]
// CHECK-LABEL: define {{.*}} @take_struct_2(i32, i32, float, float)
// CHECK:   [[V:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store i32 %0, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i32 %1, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   store float %2, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   store float %3, float* [[T0]], align 4
// CHECK:   ret void
// CHECK-LABEL: define void @test_struct_2()
// CHECK:   [[TMP:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_struct_2()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store i32 [[T1]], i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i32 [[T1]], i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_struct_2(i32 [[FIRST]], i32 [[SECOND]], float [[THIRD]], float [[FOURTH]])
// CHECK:   ret void

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
// CHECK-LABEL: define {{.*}} @return_struct_misaligned_1()
// CHECK:   [[RET:%.*]] = alloca [[REC:%.*]], align
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align
// CHECK:   @llvm.memset
// CHECK:   @llvm.memcpy
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ i32, i8 }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i8, i8* [[T0]], align
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ i32, i8 }]] undef, i32 [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i8 [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_struct_misaligned_1(i32, i8)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store i32 %0, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i8 %1, i8* [[T0]], align
// CHECK:   ret void

// Too many scalars.
typedef struct {
  int x[5];
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
// CHECK-LABEL: define {{.*}} @return_union_het_fp()
// CHECK:   [[RET:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   [[VAR:%.*]] = alloca [[REC]], align 4
// CHECK:   @llvm.memcpy
// CHECK:   @llvm.memcpy
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[RET]] to [[AGG:{ i32, i32 }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ i32, i32 }]] undef, i32 [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i32 [[SECOND]], 1
// CHECK:   ret [[UAGG]] [[T1]]
// CHECK-LABEL: define {{.*}} @take_union_het_fp(i32, i32)
// CHECK:   [[V:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store i32 %0, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i32 %1, i32* [[T0]], align 4
// CHECK:   ret void
// CHECK-LABEL: define void @test_union_het_fp()
// CHECK:   [[TMP:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG]] @return_union_het_fp()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store i32 [[T1]], i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i32 [[T1]], i32* [[T0]], align 4
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_union_het_fp(i32 [[FIRST]], i32 [[SECOND]])
// CHECK:   ret void


typedef union {
  float f1;
  float f2;
} union_hom_fp;
TEST(union_hom_fp)
// CHECK-LABEL: define void @test_union_hom_fp()
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
// CHECK-LABEL: define void @test_union_hom_fp_partial()
// CHECK:   [[TMP:%.*]] = alloca [[REC:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG:{ float, float, float, float }]] @return_union_hom_fp_partial()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG:{ float, float, float, float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_union_hom_fp_partial(float [[FIRST]], float [[SECOND]], float [[THIRD]], float [[FOURTH]])
// CHECK:   ret void

typedef union {
  struct { int x, y; } f1;
  float4 fv2;
} union_het_fpv_partial;
TEST(union_het_fpv_partial)
// CHECK-LABEL: define void @test_union_het_fpv_partial()
// CHECK:   [[TMP:%.*]] = alloca [[REC:%.*]], align 16
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC]] [[UAGG:{ i32, i32, float, float }]] @return_union_het_fpv_partial()
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG:{ i32, i32, float, float }]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store i32 [[T1]], i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i32 [[T1]], i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 2
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 3
// CHECK:   store float [[T1]], float* [[T0]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[TMP]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   [[SECOND:%.*]] = load i32, i32* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 2
// CHECK:   [[THIRD:%.*]] = load float, float* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 3
// CHECK:   [[FOURTH:%.*]] = load float, float* [[T0]], align
// CHECK:   call [[SWIFTCC]] void @take_union_het_fpv_partial(i32 [[FIRST]], i32 [[SECOND]], float [[THIRD]], float [[FOURTH]])
// CHECK:   ret void

/*****************************************************************************/
/****************************** VECTOR LEGALIZATION **************************/
/*****************************************************************************/

TEST(int4)
// CHECK-LABEL: define {{.*}} <4 x i32> @return_int4()
// CHECK-LABEL: define {{.*}} @take_int4(<4 x i32>

TEST(int8)
// CHECK-LABEL: define {{.*}} @return_int8()
// CHECK:   [[RET:%.*]] = alloca [[REC:<8 x i32>]], align 32
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
// CHECK-LABEL: define {{.*}} @take_int8(<4 x i32>, <4 x i32>)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store <4 x i32> %1, <4 x i32>* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define void @test_int8()
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
// CHECK:   [[RET:%.*]] = alloca [[REC:<5 x i32>]], align 32
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
// CHECK-LABEL: define {{.*}} @take_int5(<4 x i32>, i32)
// CHECK:   [[V:%.*]] = alloca [[REC]], align
// CHECK:   [[CAST_TMP:%.*]] = bitcast [[REC]]* [[V]] to [[AGG]]*
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 0
// CHECK:   store <4 x i32> %0, <4 x i32>* [[T0]], align
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[CAST_TMP]], i32 0, i32 1
// CHECK:   store i32 %1, i32* [[T0]], align
// CHECK:   ret void
// CHECK-LABEL: define void @test_int5()
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
// CHECK-LABEL: define {{.*}} @take_misaligned_int3(i32, i32, i32, i32)
