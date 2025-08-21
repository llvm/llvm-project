// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -fobjc-exceptions -fexceptions -debug-info-kind=line-tables-only -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_STRONG:.*]] = type { i32, ptr }
// CHECK: %[[STRUCT_WEAK:.*]] = type { i32, ptr }

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
// CHECK: store [2 x i64] %[[CALL]], ptr %[[AGG_TMP]], align 8
// CHECK: invoke [2 x i64] @genStrong()

// CHECK: call void @calleeStrong([2 x i64] %{{.*}}, [2 x i64] %{{.*}})
// CHECK-NEXT: ret void

// CHECK: landingpad { ptr, i32 }
// CHECK: call void @__destructor_8_s8(ptr %[[AGG_TMP]]){{.*}}, !dbg [[ARTIFICIAL_LOC_1:![0-9]+]]
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
// CHECK: call void @genWeak(ptr dead_on_unwind writable sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_TMP]])
// CHECK: invoke void @genWeak(ptr dead_on_unwind writable sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_TMP1]])

// CHECK: call void @calleeWeak(ptr noundef %[[AGG_TMP]], ptr noundef %[[AGG_TMP1]])
// CHECK: ret void

// CHECK: landingpad { ptr, i32 }
// CHECK: call void @__destructor_8_w8(ptr %[[AGG_TMP]]){{.*}}, !dbg [[ARTIFICIAL_LOC_2:![0-9]+]]
// CHECK: br label

// CHECK: resume

// CHECK: define{{.*}} void @__destructor_8_w8({{.*}} !dbg ![[DTOR_SP:.*]] {
// CHECK: load ptr, ptr {{.*}}, !dbg ![[DTOR_LOC:.*]]

Weak genWeak(void);
void calleeWeak(Weak, Weak);

void testWeakException(void) {
  calleeWeak(genWeak(), genWeak());
}

// CHECK-DAG: [[ARTIFICIAL_LOC_1]] = !DILocation(line: 0
// CHECK-DAG: [[ARTIFICIAL_LOC_2]] = !DILocation(line: 0
// CHECK: ![[DTOR_SP]] = distinct !DISubprogram(linkageName: "__destructor_8_w8",
// CHECK: ![[DTOR_LOC]] = !DILocation(line: 0, scope: ![[DTOR_SP]])
