// Test debug info for intermediate value of a chained pointer deferencing
// expression when the flag -fdebug-info-for-pointer-type is enabled.
// RUN: %clang_cc1 %s -fdebug-info-for-profiling -debug-info-kind=constructor -emit-llvm -o - | FileCheck %s

class A {
public:
  int i;
  char c;
  void *p;
  int arr[3];
};

class B {
public:
  A* a;
};

class C {
public:
  B* b;
  A* a;
  A arr[10];
};

// CHECK-LABEL: define dso_local noundef i32 @{{.*}}func1{{.*}}(
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds %class.B, ptr {{%.*}}, i32 0, i32 0, !dbg [[DBG1:![0-9]+]]
// CHECK-NEXT:    [[A:%.*]] = load ptr, ptr [[A_ADDR]], align {{.*}}, !dbg [[DBG1]]
// CHECK-NEXT:    [[PSEUDO1:%.*]] = alloca ptr, align {{.*}}, !dbg [[DBG1]]
// CHECK-NEXT:    store ptr [[A]], ptr [[PSEUDO1]], align {{.*}}, !dbg [[DBG1]]
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[PSEUDO1]], metadata [[META1:![0-9]+]], metadata !DIExpression()), !dbg [[DBG1]]
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PSEUDO1]], align {{.*}}, !dbg [[DBG1]]
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds %class.A, ptr [[TMP1]], i32 0, i32 0,
int func1(B *b) {
  return b->a->i;
}

// Should generate a pseudo variable when pointer is type-casted.
// CHECK-LABEL: define dso_local noundef ptr @{{.*}}func2{{.*}}(
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[B_ADDR:%.*]], metadata [[META2:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    [[B:%.*]] = load ptr, ptr [[B_ADDR]],
// CHECK-NEXT:    [[PSEUDO1:%.*]] = alloca ptr,
// CHECK-NEXT:    store ptr [[B]], ptr [[PSEUDO1]],
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[PSEUDO1]], metadata [[META3:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PSEUDO1]],
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds %class.B, ptr [[TMP1]], i32 0,
A* func2(void *b) {
  return ((B*)b)->a;
}

// Should not generate pseudo variable in this case.
// CHECK-LABEL: define dso_local noundef i32 @{{.*}}func3{{.*}}(
// CHECK:    call void @llvm.dbg.declare(metadata ptr [[B_ADDR:%.*]], metadata [[META4:![0-9]+]], metadata !DIExpression())
// CHECK:    call void @llvm.dbg.declare(metadata ptr [[LOCAL1:%.*]], metadata [[META5:![0-9]+]], metadata !DIExpression())
// CHECK-NOT: call void @llvm.dbg.declare(metadata ptr
int func3(B *b) {
  A *local1 = b->a;
  return local1->i;
}

// CHECK-LABEL: define dso_local noundef signext i8 @{{.*}}func4{{.*}}(
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds %class.C, ptr {{%.*}}, i32 0, i32 1
// CHECK-NEXT:    [[A:%.*]] = load ptr, ptr [[A_ADDR]],
// CHECK-NEXT:    [[PSEUDO1:%.*]] = alloca ptr,
// CHECK-NEXT:    store ptr [[A]], ptr [[PSEUDO1]],
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[PSEUDO1]], metadata [[META6:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PSEUDO1]],
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds %class.A, ptr [[TMP1]], i32 0, i32 0,
// CHECK:         [[CALL:%.*]] = call noundef ptr @{{.*}}foo{{.*}}(
// CHECK-NEXT:    [[PSEUDO2:%.*]] = alloca ptr,
// CHECK-NEXT:    store ptr [[CALL]], ptr [[PSEUDO2]]
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[PSEUDO2]], metadata [[META6]], metadata !DIExpression())
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[PSEUDO2]]
// CHECK-NEXT:    [[I1:%.*]] = getelementptr inbounds %class.A, ptr [[TMP2]], i32 0, i32 1
char func4(C *c) {
  extern A* foo(int x);
  return foo(c->a->i)->c;
}

// CHECK-LABEL: define dso_local noundef signext i8 @{{.*}}func5{{.*}}(
// CHECK:         call void @llvm.dbg.declare(metadata ptr {{%.*}}, metadata [[META7:![0-9]+]], metadata !DIExpression())
// CHECK:         call void @llvm.dbg.declare(metadata ptr {{%.*}}, metadata [[META8:![0-9]+]], metadata !DIExpression())
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds %class.A, ptr {{%.*}}, i64 {{%.*}},
// CHECK-NEXT:    [[PSEUDO1:%.*]] = alloca ptr,
// CHECK-NEXT:    store ptr [[A_ADDR]], ptr [[PSEUDO1]],
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[PSEUDO1]], metadata [[META9:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PSEUDO1]],
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds %class.A, ptr [[TMP1]], i32 0, i32 1,
char func5(void *arr, int n) {
  return ((A*)arr)[n].c;
}

// CHECK-LABEL: define dso_local noundef i32 @{{.*}}func6{{.*}}(
// CHECK:         call void @llvm.dbg.declare(metadata ptr {{%.*}}, metadata [[META10:![0-9]+]], metadata !DIExpression())
// CHECK:         call void @llvm.dbg.declare(metadata ptr {{%.*}}, metadata [[META11:![0-9]+]], metadata !DIExpression())
int func6(B &b) {
  return reinterpret_cast<A&>(b).i;
}

// CHECK-DAG: [[META_A:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A",
// CHECK-DAG: [[META_AP:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[META_A]],
// CHECK-DAG: [[META_B:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B",
// CHECK-DAG: [[META_BP:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[META_B]],
// CHECK-DAG: [[META_C:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C",
// CHECK-DAG: [[META_CP:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[META_C]],
// CHECK-DAG: [[META_VP:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null,
// CHECK-DAG: [[META_I32:![0-9]+]] = !DIBasicType(name: "int", size: 32,
// CHECK-DAG: [[META_BR:![0-9]+]] = !DIDerivedType(tag: DW_TAG_reference_type, baseType: [[META_B]],

// CHECK-DAG: [[DBG1]] = !DILocation(line: 34, column: 13,
// CHECK-DAG: [[META1]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META2]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 46, type: [[META_VP]])
// CHECK-DAG: [[META3]] = !DILocalVariable(scope: {{.*}}, type: [[META_BP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META4]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 55, type: [[META_BP]])
// CHECK-DAG: [[META5]] = !DILocalVariable(name: "local1", scope: {{.*}}, file: {{.*}}, line: 56, type: [[META_AP]])
// CHECK-DAG: [[META6]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META7]] = !DILocalVariable(name: "arr", arg: 1, scope: {{.*}}, file: {{.*}}, line: 88, type: [[META_VP]])
// CHECK-DAG: [[META8]] = !DILocalVariable(name: "n", arg: 2, scope: {{.*}}, file: {{.*}}, line: 88, type: [[META_I32]])
// CHECK-DAG: [[META9]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META10]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 95, type: [[META_BR]])
// CHECK-DAG: [[META11]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
