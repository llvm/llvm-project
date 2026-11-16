// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv-pointer -fclangir -emit-cir %s -o %t-wrapv.cir
// RUN: FileCheck --check-prefix=WRAPV-CIR --input-file=%t-wrapv.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv-pointer -fclangir -emit-llvm %s -o %t-wrapv-cir.ll
// RUN: FileCheck --check-prefixes=WRAPV-LLVM,WRAPV-LLVMCIR --input-file=%t-wrapv-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv-pointer -emit-llvm %s -o %t-wrapv.ll
// RUN: FileCheck --check-prefixes=WRAPV-LLVM,WRAPV-OGCG --input-file=%t-wrapv.ll %s

int signed_index(int i) {
  int a[4] = {0};
  return a[i];
}

// CIR-LABEL: cir.func {{.*}} @signed_index(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:         cir.get_element inbounds %[[A]][%[[IDX]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @signed_index(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         %[[IDX:.*]] = sext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[IDX]]

// WRAPV-CIR-LABEL: cir.func {{.*}} @signed_index(
// WRAPV-CIR:         cir.get_element %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// WRAPV-LLVM-LABEL: define dso_local i32 @signed_index(
// WRAPV-LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// WRAPV-LLVM:         %[[IDX:.*]] = sext i32 %{{.*}} to i64
// WRAPV-LLVM:         getelementptr [4 x i32], ptr %[[A]],
// WRAPV-LLVMCIR-SAME:   i32 0,
// WRAPV-OGCG-SAME:      i64 0,
// WRAPV-LLVM-SAME:      i64 %[[IDX]]

int unsigned_index(unsigned u) {
  int a[4] = {0};
  return a[u];
}

// CIR-LABEL: cir.func {{.*}} @unsigned_index(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.*}} : !u32i -> !u64i
// CIR:         cir.get_element inbounds nuw %[[A]][%[[IDX]] : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @unsigned_index(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         %[[IDX:.*]] = zext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds nuw [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[IDX]]

// WRAPV-CIR-LABEL: cir.func {{.*}} @unsigned_index(
// WRAPV-CIR:         cir.get_element %{{.*}}[%{{.*}} : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// WRAPV-LLVM-LABEL: define dso_local i32 @unsigned_index(
// WRAPV-LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// WRAPV-LLVM:         %[[IDX:.*]] = zext i32 %{{.*}} to i64
// WRAPV-LLVM:         getelementptr [4 x i32], ptr %[[A]],
// WRAPV-LLVMCIR-SAME:   i32 0,
// WRAPV-OGCG-SAME:      i64 0,
// WRAPV-LLVM-SAME:      i64 %[[IDX]]

int constant_index(void) {
  int a[4] = {0};
  return a[1];
}

// CIR-LABEL: cir.func {{.*}} @constant_index(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:         cir.get_element inbounds %[[A]][%[[ONE]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @constant_index(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         getelementptr inbounds [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 1

int unsigned_constant_index(void) {
  int a[4] = {0};
  return a[1u];
}

// CIR-LABEL: cir.func {{.*}} @unsigned_constant_index(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         cir.get_element inbounds nuw %[[A]][%[[ONE]] : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @unsigned_constant_index(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         getelementptr inbounds nuw [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 1

enum SignedIdx { NegOne = -1, SZero, SOne };
enum UnsignedIdx { UZero, UOne };

int enum_index(enum SignedIdx s, enum UnsignedIdx u) {
  int a[4] = {0};
  return a[s] + a[u];
}

// CIR-LABEL: cir.func {{.*}} @enum_index(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[S:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:         cir.get_element inbounds %[[A]][%[[S]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CIR:         %[[U:.*]] = cir.cast integral %{{.*}} : !u32i -> !u64i
// CIR:         cir.get_element inbounds nuw %[[A]][%[[U]] : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @enum_index(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         %[[S:.*]] = sext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[S]]
// LLVM:         %[[U:.*]] = zext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds nuw [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[U]]

int reversed_subscript(int i, unsigned u) {
  int a[4] = {0};
  return i[a] + u[a];
}

// CIR-LABEL: cir.func {{.*}} @reversed_subscript(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[I:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:         cir.get_element inbounds %[[A]][%[[I]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CIR:         %[[U:.*]] = cir.cast integral %{{.*}} : !u32i -> !u64i
// CIR:         cir.get_element inbounds nuw %[[A]][%[[U]] : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @reversed_subscript(
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 16
// LLVM:         %[[I:.*]] = sext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[I]]
// LLVM:         %[[U:.*]] = zext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds nuw [4 x i32], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[U]]

int two_dimensions(int i, unsigned j) {
  int a[3][4] = {{0}};
  return a[i][j];
}

// CIR-LABEL: cir.func {{.*}} @two_dimensions(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>>
// CIR:         %[[I:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:         %[[ROW:.*]] = cir.get_element inbounds %[[A]][%[[I]] : !s64i] : !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>> -> !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         %[[J:.*]] = cir.cast integral %{{.*}} : !u32i -> !u64i
// CIR:         cir.get_element inbounds nuw %[[ROW]][%[[J]] : !u64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @two_dimensions(
// LLVM:         %[[A:.*]] = alloca [3 x [4 x i32]], align 16
// LLVM:         %[[I:.*]] = sext i32 %{{.*}} to i64
// LLVM:         %[[ROW:.*]] = getelementptr inbounds [3 x [4 x i32]], ptr %[[A]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[I]]
// LLVM:         %[[J:.*]] = zext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds nuw [4 x i32], ptr %[[ROW]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[J]]

int pointer_to_2d(int (*p)[3][4], int i, int j) { return (*p)[i][j]; }

// CIR-LABEL: cir.func {{.*}} @pointer_to_2d(
// CIR:         %[[ROW:.*]] = cir.get_element inbounds %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>> -> !cir.ptr<!cir.array<!s32i x 4>>
// CIR:         cir.get_element inbounds %[[ROW]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @pointer_to_2d(
// LLVM:         %[[ROW:.*]] = getelementptr inbounds [3 x [4 x i32]], ptr
// LLVM:         getelementptr inbounds [4 x i32], ptr %[[ROW]],

struct Flex {
  int n;
  int a[];
};

int flexible_member(struct Flex *s, int i) { return s->a[i]; }

// CIR-LABEL: cir.func {{.*}} @flexible_member(
// CIR:         %[[MEMBER:.*]] = cir.get_member %{{.*}}[1] {name = "a"} : !cir.ptr<!rec_Flex> -> !cir.ptr<!cir.array<!s32i x 0>>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:         cir.get_element inbounds %[[MEMBER]][%[[IDX]] : !s64i] : !cir.ptr<!cir.array<!s32i x 0>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @flexible_member(
// LLVM:         %[[MEMBER:.*]] = getelementptr inbounds nuw %struct.Flex, ptr %{{.*}}, i32 0, i32 1
// LLVM:         %[[IDX:.*]] = sext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds [0 x i32], ptr %[[MEMBER]],
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[IDX]]

static const int debruijn[32] = {
    0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};

int count_trailing_zeros(unsigned x) {
  return debruijn[((x & -x) * 0x077CB531u) >> 27];
}

// CIR-LABEL: cir.func {{.*}} @count_trailing_zeros(
// CIR:         %[[TABLE:.*]] = cir.get_global @debruijn : !cir.ptr<!cir.array<!s32i x 32>>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.*}} : !u32i -> !u64i
// CIR:         cir.get_element inbounds nuw %[[TABLE]][%[[IDX]] : !u64i] : !cir.ptr<!cir.array<!s32i x 32>> -> !cir.ptr<!s32i>

// LLVM-LABEL: define dso_local i32 @count_trailing_zeros(
// LLVM:         %[[IDX:.*]] = zext i32 %{{.*}} to i64
// LLVM:         getelementptr inbounds nuw [32 x i32], ptr @debruijn,
// LLVMCIR-SAME:   i32 0,
// OGCG-SAME:      i64 0,
// LLVM-SAME:      i64 %[[IDX]]
