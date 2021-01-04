// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -fobjc-exceptions -fexceptions -debug-info-kind=line-tables-only -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_STRONG:.*]] = type { i32, i8* }
// CHECK: %[[STRUCT_WEAK:.*]] = type { i32, i8* }

typedef struct {
  int i;
  id f1;
} Strong;

typedef struct {
  int i;
  __weak id f1;
} Weak;

// CHECK: define{{.*}} void @testStrongException()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[AGG_TMP1:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[CALL:.*]] = call [2 x i64] @genStrong()
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONG]]* %[[AGG_TMP]] to [2 x i64]*
// CHECK: store [2 x i64] %[[CALL]], [2 x i64]* %[[V0]], align 8
// CHECK: invoke [2 x i64] @genStrong()

// CHECK: call void @calleeStrong([2 x i64] %{{.*}}, [2 x i64] %{{.*}})
// CHECK-NEXT: ret void

// CHECK: landingpad { i8*, i32 }
// CHECK: %[[V9:.*]] = bitcast %[[STRUCT_STRONG]]* %[[AGG_TMP]] to i8**{{.*}}, !dbg [[ARTIFICIAL_LOC_1:![0-9]+]]
// CHECK: call void @__destructor_8_s8(i8** %[[V9]]){{.*}}, !dbg [[ARTIFICIAL_LOC_1]]
// CHECK: br label

// CHECK: resume

Strong genStrong(void);
void calleeStrong(Strong, Strong);

void testStrongException(void) {
  calleeStrong(genStrong(), genStrong());
}

// CHECK: define{{.*}} void @testWeakException()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_WEAK]], align 8
// CHECK: %[[AGG_TMP1:.*]] = alloca %[[STRUCT_WEAK]], align 8
// CHECK: call void @genWeak(%[[STRUCT_WEAK]]* sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_TMP]])
// CHECK: invoke void @genWeak(%[[STRUCT_WEAK]]* sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_TMP1]])

// CHECK: call void @calleeWeak(%[[STRUCT_WEAK]]* %[[AGG_TMP]], %[[STRUCT_WEAK]]* %[[AGG_TMP1]])
// CHECK: ret void

// CHECK: landingpad { i8*, i32 }
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_TMP]] to i8**{{.*}}, !dbg [[ARTIFICIAL_LOC_2:![0-9]+]]
// CHECK: call void @__destructor_8_w8(i8** %[[V3]]){{.*}}, !dbg [[ARTIFICIAL_LOC_2]]
// CHECK: br label

// CHECK: resume

Weak genWeak(void);
void calleeWeak(Weak, Weak);

void testWeakException(void) {
  calleeWeak(genWeak(), genWeak());
}

// CHECK-DAG: [[ARTIFICIAL_LOC_1]] = !DILocation(line: 0
// CHECK-DAG: [[ARTIFICIAL_LOC_2]] = !DILocation(line: 0
