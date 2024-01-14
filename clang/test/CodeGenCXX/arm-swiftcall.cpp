// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -o - %s -Wno-return-type-c-linkage -std=c++03 | FileCheck %s -check-prefixes=CHECK

// This isn't really testing anything ARM-specific; it's just a convenient
// 32-bit platform.

#define SWIFTCALL __attribute__((swiftcall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))

/*****************************************************************************/
/********************************** LOWERING *********************************/
/*****************************************************************************/

#define TEST(TYPE)                                  \
  extern "C" SWIFTCALL TYPE return_##TYPE(void) {   \
    TYPE result = {};                               \
    return result;                                  \
  }                                                 \
  extern "C" SWIFTCALL void take_##TYPE(TYPE v) {   \
  }                                                 \
  extern "C" void test_##TYPE() {                   \
    take_##TYPE(return_##TYPE());                   \
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

// This is only properly testable in C++ because it relies on empty structs
// actually taking up space in a structure without requiring any extra data
// to be passed.
typedef struct {
  int x;
  struct_empty padding[2];
  char c1;
  float f0;
  float f1;
} struct_1;
TEST(struct_1);
// CHECK-LABEL: define {{.*}} @return_struct_1()
// CHECK:   [[RET:%.*]] = alloca [[REC:%.*]], align 4
// CHECK:   @llvm.memset
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG:{ i32, \[2 x i8\], i8, \[1 x i8\], float, float }]], ptr [[RET]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[RET]], i32 0, i32 2
// CHECK:   [[SECOND:%.*]] = load i8, ptr [[T0]], align 2
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[RET]], i32 0, i32 4
// CHECK:   [[THIRD:%.*]] = load float, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[RET]], i32 0, i32 5
// CHECK:   [[FOURTH:%.*]] = load float, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = insertvalue [[UAGG:{ i32, i8, float, float }]] poison, i32 [[FIRST]], 0
// CHECK:   [[T1:%.*]] = insertvalue [[UAGG]] [[T0]], i8 [[SECOND]], 1
// CHECK:   [[T2:%.*]] = insertvalue [[UAGG]] [[T1]], float [[THIRD]], 2
// CHECK:   [[T3:%.*]] = insertvalue [[UAGG]] [[T2]], float [[FOURTH]], 3
// CHECK:   ret [[UAGG]] [[T3]]
// CHECK-LABEL: define {{.*}} @take_struct_1(i32 %0, i8 %1, float %2, float %3)
// CHECK:   [[V:%.*]] = alloca [[REC]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[V]], i32 0, i32 0
// CHECK:   store i32 %0, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[V]], i32 0, i32 2
// CHECK:   store i8 %1, ptr [[T0]], align 2
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[V]], i32 0, i32 4
// CHECK:   store float %2, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[V]], i32 0, i32 5
// CHECK:   store float %3, ptr [[T0]], align 4
// CHECK:   ret void
// CHECK-LABEL: define{{.*}} void @test_struct_1()
// CHECK:   [[TMP:%.*]] = alloca [[REC]], align 4
// CHECK:   [[CALL:%.*]] = call [[SWIFTCC:swiftcc]] [[UAGG]] @return_struct_1()
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 0
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 0
// CHECK:   store i32 [[T1]], ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 2
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 1
// CHECK:   store i8 [[T1]], ptr [[T0]], align 2
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 4
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 2
// CHECK:   store float [[T1]], ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 5
// CHECK:   [[T1:%.*]] = extractvalue [[UAGG]] [[CALL]], 3
// CHECK:   store float [[T1]], ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 0
// CHECK:   [[FIRST:%.*]] = load i32, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 2
// CHECK:   [[SECOND:%.*]] = load i8, ptr [[T0]], align 2
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 4
// CHECK:   [[THIRD:%.*]] = load float, ptr [[T0]], align 4
// CHECK:   [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TMP]], i32 0, i32 5
// CHECK:   [[FOURTH:%.*]] = load float, ptr [[T0]], align 4
// CHECK:   call [[SWIFTCC]] void @take_struct_1(i32 [[FIRST]], i8 [[SECOND]], float [[THIRD]], float [[FOURTH]])
// CHECK:   ret void

struct struct_indirect_1 {
  int x;
  ~struct_indirect_1();
};
TEST(struct_indirect_1)

// CHECK-LABEL: define {{.*}} void @return_struct_indirect_1({{.*}} dead_on_unwind noalias writable sret

// Should not be byval.
// CHECK-LABEL: define {{.*}} void @take_struct_indirect_1(ptr noundef{{( %.*)?}})

// Do a simple standalone test here of a function definition to ensure that
// we don't have problems due to failure to eagerly synthesize a copy
// constructor declaration.
class struct_trivial {
  int x;
};
// CHECK-LABEL: define{{.*}} swiftcc void @test_struct_trivial(i32{{( %.*)?}})
extern "C" SWIFTCALL
void test_struct_trivial(struct_trivial triv) {}
