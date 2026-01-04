// Test debug info for intermediate value of a chained pointer deferencing
// expression when the flag -fdebug-info-for-pointer-type is enabled.
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu %s -fdebug-info-for-profiling -debug-info-kind=constructor -o - | FileCheck %s

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
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds nuw %class.B, ptr {{%.*}}, i32 0, i32 0, !dbg [[DBG1:![0-9]+]]
// CHECK-NEXT:    [[A:%.*]] = load ptr, ptr [[A_ADDR]], align {{.*}}, !dbg [[DBG1]]
// CHECK-NEXT:      #dbg_value(ptr [[A]], [[META1:![0-9]+]], !DIExpression(), [[DBG1]])
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds nuw %class.A, ptr [[A]], i32 0, i32 0,
int func1(B *b) {
  return b->a->i;
}

// Should generate a pseudo variable when pointer is type-casted.
// CHECK-LABEL: define dso_local noundef ptr @{{.*}}func2{{.*}}(
// CHECK:           #dbg_declare(ptr [[B_ADDR:%.*]], [[META2:![0-9]+]], !DIExpression(),
// CHECK-NEXT:    [[B:%.*]] = load ptr, ptr [[B_ADDR]],
// CHECK-NEXT:      #dbg_value(ptr [[B]], [[META3:![0-9]+]], !DIExpression(),
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds nuw %class.B, ptr [[B]], i32 0,
A* func2(void *b) {
  return ((B*)b)->a;
}

// Should not generate pseudo variable in this case.
// CHECK-LABEL: define dso_local noundef i32 @{{.*}}func3{{.*}}(
// CHECK:      #dbg_declare(ptr [[B_ADDR:%.*]], [[META4:![0-9]+]], !DIExpression(),
// CHECK:      #dbg_declare(ptr [[LOCAL1:%.*]], [[META5:![0-9]+]], !DIExpression(),
// CHECK-NOT:  #dbg_value(ptr
int func3(B *b) {
  A *local1 = b->a;
  return local1->i;
}

// CHECK-LABEL: define dso_local noundef signext i8 @{{.*}}func4{{.*}}(
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds nuw %class.C, ptr {{%.*}}, i32 0, i32 1
// CHECK-NEXT:    [[A:%.*]] = load ptr, ptr [[A_ADDR]],
// CHECK-NEXT:      #dbg_value(ptr [[A]], [[META6:![0-9]+]], !DIExpression(),
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds nuw %class.A, ptr [[A]], i32 0, i32 0,
// CHECK:         [[CALL:%.*]] = call noundef ptr @{{.*}}foo{{.*}}(
// CHECK-NEXT:      #dbg_value(ptr [[CALL]], [[META6]], !DIExpression(),
// CHECK-NEXT:    [[I1:%.*]] = getelementptr inbounds nuw %class.A, ptr [[CALL]], i32 0, i32 1
char func4(C *c) {
  extern A* foo(int x);
  return foo(c->a->i)->c;
}

// CHECK-LABEL: define dso_local noundef signext i8 @{{.*}}func5{{.*}}(
// CHECK:           #dbg_declare(ptr {{%.*}}, [[META7:![0-9]+]], !DIExpression(),
// CHECK:           #dbg_declare(ptr {{%.*}}, [[META8:![0-9]+]], !DIExpression(),
// CHECK:         [[A_ADDR:%.*]] = getelementptr inbounds %class.A, ptr {{%.*}}, i64 {{%.*}},
// CHECK-NEXT:      #dbg_value(ptr [[A_ADDR]], [[META9:![0-9]+]], !DIExpression(),
// CHECK-NEXT:    {{%.*}} = getelementptr inbounds nuw %class.A, ptr [[A_ADDR]], i32 0, i32 1,
char func5(void *arr, int n) {
  return ((A*)arr)[n].c;
}

// CHECK-LABEL: define dso_local noundef i32 @{{.*}}func6{{.*}}(
// CHECK:           #dbg_declare(ptr {{%.*}}, [[META10:![0-9]+]], !DIExpression(),
// CHECK:           #dbg_value(ptr {{%.*}}, [[META11:![0-9]+]], !DIExpression(),
int func6(B &b) {
  return reinterpret_cast<A&>(b).i;
}

// CHECK-LABEL: define dso_local noundef i32 @{{.*}}global{{.*}}(
// CHECK:         [[GA:%.*]] = load ptr, ptr @ga
// CHECK-NEXT:      #dbg_value(ptr [[GA]], [[META12:![0-9]+]], !DIExpression(),
A *ga;
int global() {
  return ga->i;
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

// CHECK-DAG: [[DBG1]] = !DILocation(line: 31, column: 13,
// CHECK-DAG: [[META1]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META2]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 40, type: [[META_VP]])
// CHECK-DAG: [[META3]] = !DILocalVariable(scope: {{.*}}, type: [[META_BP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META4]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 49, type: [[META_BP]])
// CHECK-DAG: [[META5]] = !DILocalVariable(name: "local1", scope: {{.*}}, file: {{.*}}, line: 50, type: [[META_AP]])
// CHECK-DAG: [[META6]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META7]] = !DILocalVariable(name: "arr", arg: 1, scope: {{.*}}, file: {{.*}}, line: 73, type: [[META_VP]])
// CHECK-DAG: [[META8]] = !DILocalVariable(name: "n", arg: 2, scope: {{.*}}, file: {{.*}}, line: 73, type: [[META_I32]])
// CHECK-DAG: [[META9]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META10]] = !DILocalVariable(name: "b", arg: 1, scope: {{.*}}, file: {{.*}}, line: 80, type: [[META_BR]])
// CHECK-DAG: [[META11]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
// CHECK-DAG: [[META12]] = !DILocalVariable(scope: {{.*}}, type: [[META_AP]], flags: DIFlagArtificial)
