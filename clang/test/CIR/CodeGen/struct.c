// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
} bar;

struct Foo {
  int a;
  char b;
  struct Bar z;
};

// Recursive type
typedef struct Node {
  struct Node* next;
} NodeStru;

void baz(void) {
  struct Bar b;
  struct Foo f;
}

// CHECK-DAG: !ty_Node = !cir.struct<struct "Node" {!cir.ptr<!cir.struct<struct "Node">>} #cir.record.decl.ast>
// CHECK-DAG: !ty_Bar = !cir.struct<struct "Bar" {!cir.int<s, 32>, !cir.int<s, 8>}>
// CHECK-DAG: !ty_Foo = !cir.struct<struct "Foo" {!cir.int<s, 32>, !cir.int<s, 8>, !cir.struct<struct "Bar" {!cir.int<s, 32>, !cir.int<s, 8>}>}>
// CHECK-DAG: !ty_SLocal = !cir.struct<struct "SLocal" {!cir.int<s, 32>}>
// CHECK-DAG: !ty_SLocal2E0_ = !cir.struct<struct "SLocal.0" {!cir.float}>
//  CHECK-DAG: module {{.*}} {
     // CHECK:   cir.func @baz()
// CHECK-NEXT:     %0 = cir.alloca !ty_Bar, !cir.ptr<!ty_Bar>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !ty_Foo, !cir.ptr<!ty_Foo>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }

void shouldConstInitStructs(void) {
// CHECK: cir.func @shouldConstInitStructs
  struct Foo f = {1, 2, {3, 4}};
  // CHECK: %[[#V0:]] = cir.alloca !ty_Foo, !cir.ptr<!ty_Foo>, ["f"] {alignment = 4 : i64}
  // CHECK: %[[#V1:]] = cir.const #cir.const_struct<{#cir.int<1> : !s32i, #cir.int<2> : !s8i, #cir.const_struct<{#cir.int<3> : !s32i, #cir.int<4> : !s8i}> : !ty_Bar}> : !ty_Foo
  // CHECK: cir.store %[[#V1]], %[[#V0]] : !ty_Foo, !cir.ptr<!ty_Foo>
}

// Should zero-initialize uninitialized global structs.
struct S {
  int a,b;
} s;
// CHECK-DAG: cir.global external @s = #cir.zero : !ty_S

// Should initialize basic global structs.
struct S1 {
  int a;
  float f;
  int *p;
} s1 = {1, .1, 0};
// CHECK-DAG: cir.global external @s1 = #cir.const_struct<{#cir.int<1> : !s32i, #cir.fp<1.000000e-01> : !cir.float, #cir.ptr<null> : !cir.ptr<!s32i>}> : !ty_S1_

// Should initialize global nested structs.
struct S2 {
  struct S2A {
    int a;
  } s2a;
} s2 = {{1}};
// CHECK-DAG: cir.global external @s2 = #cir.const_struct<{#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_S2A}> : !ty_S2_

// Should initialize global arrays of structs.
struct S3 {
  int a;
} s3[3] = {{1}, {2}, {3}};
// CHECK-DAG: cir.global external @s3 = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_S3_, #cir.const_struct<{#cir.int<2> : !s32i}> : !ty_S3_, #cir.const_struct<{#cir.int<3> : !s32i}> : !ty_S3_]> : !cir.array<!ty_S3_ x 3>

void shouldCopyStructAsCallArg(struct S1 s) {
// CHECK-DAG: cir.func @shouldCopyStructAsCallArg
  shouldCopyStructAsCallArg(s);
  // CHECK-DAG: %[[#LV:]] = cir.load %{{.+}} : !cir.ptr<!ty_S1_>, !ty_S1_
  // CHECK-DAG: cir.call @shouldCopyStructAsCallArg(%[[#LV]]) : (!ty_S1_) -> ()
}

struct Bar shouldGenerateAndAccessStructArrays(void) {
  struct Bar s[1] = {{3, 4}};
  return s[0];
}
// CHECK-DAG: cir.func @shouldGenerateAndAccessStructArrays
// CHECK-DAG: %[[#STRIDE:]] = cir.const #cir.int<0> : !s32i
// CHECK-DAG: %[[#DARR:]] = cir.cast(array_to_ptrdecay, %{{.+}} : !cir.ptr<!cir.array<!ty_Bar x 1>>), !cir.ptr<!ty_Bar>
// CHECK-DAG: %[[#ELT:]] = cir.ptr_stride(%[[#DARR]] : !cir.ptr<!ty_Bar>, %[[#STRIDE]] : !s32i), !cir.ptr<!ty_Bar>
// CHECK-DAG: cir.copy %[[#ELT]] to %{{.+}} : !cir.ptr<!ty_Bar>

// CHECK-DAG: cir.func @local_decl
// CHECK-DAG: {{%.}} = cir.alloca !ty_Local, !cir.ptr<!ty_Local>, ["a"]
void local_decl(void) {
  struct Local {
    int i;
  };
  struct Local a;
}

// CHECK-DAG: cir.func @useRecursiveType
// CHECK-DAG: cir.get_member {{%.}}[0] {name = "next"} : !cir.ptr<!ty_Node> -> !cir.ptr<!cir.ptr<!ty_Node>>
void useRecursiveType(NodeStru* a) {
  a->next = 0;
}

// CHECK-DAG: cir.alloca !ty_SLocal, !cir.ptr<!ty_SLocal>, ["loc", init] {alignment = 4 : i64}
// CHECK-DAG: cir.scope {
// CHECK-DAG:   cir.alloca !ty_SLocal2E0_, !cir.ptr<!ty_SLocal2E0_>, ["loc", init] {alignment = 4 : i64}
void local_structs(int a, float b) {
  struct SLocal { int x; };
  struct SLocal loc = {a};
  {
    struct SLocal { float y; };
    struct SLocal loc = {b};
  }   
}
