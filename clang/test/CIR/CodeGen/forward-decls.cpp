// RUN: split-file %s %t


//--- incomplete_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/incomplete_struct -o %t/incomplete_struct.cir
// RUN: FileCheck %s --input-file=%t/incomplete_struct.cir --check-prefix=CHECK1

// Forward declaration of the record is never defined, so it is created as
// an incomplete struct in CIR and will remain as such.

// CHECK1: ![[INC_STRUCT:.+]] = !cir.struct<struct "IncompleteStruct" incomplete>
struct IncompleteStruct;
// CHECK1: testIncompleteStruct(%arg0: !cir.ptr<![[INC_STRUCT]]>
void testIncompleteStruct(struct IncompleteStruct *s) {};



//--- mutated_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/mutated_struct -o %t/mutated_struct.cir
// RUN: FileCheck %s --input-file=%t/mutated_struct.cir --check-prefix=CHECK2

// Foward declaration of the struct is followed by usage, then definition.
// This means it will initially be created as incomplete, then completed.

// CHECK2: ![[COMPLETE:.+]] = !cir.struct<struct "ForwardDeclaredStruct" {!cir.int<s, 32>} #cir.record.decl.ast>
// CHECK2: testForwardDeclaredStruct(%arg0: !cir.ptr<![[COMPLETE]]>
struct ForwardDeclaredStruct;
void testForwardDeclaredStruct(struct ForwardDeclaredStruct *fds) {};
struct ForwardDeclaredStruct {
  int testVal;
};



//--- recursive_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/recursive_struct -o %t/recursive_struct.cir
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t/recursive_struct.cir %s

// Struct is initially forward declared since the self-reference is generated
// first. Then, once the type is fully generated, it is completed.

// CHECK3: ![[STRUCT:.+]] = !cir.struct<struct "RecursiveStruct" {!cir.int<s, 32>, !cir.ptr<!cir.struct<struct "RecursiveStruct">>} #cir.record.decl.ast>
struct RecursiveStruct {
  int value;
  struct RecursiveStruct *next;
};
// CHECK3: testRecursiveStruct(%arg0: !cir.ptr<![[STRUCT]]>
void testRecursiveStruct(struct RecursiveStruct *arg) {
  // CHECK3: %[[#NEXT:]] = cir.get_member %{{.+}}[1] {name = "next"} : !cir.ptr<![[STRUCT]]> -> !cir.ptr<!cir.ptr<![[STRUCT]]>>
  // CHECK3: %[[#DEREF:]] = cir.load %[[#NEXT]] : cir.ptr <!cir.ptr<![[STRUCT]]>>, !cir.ptr<![[STRUCT]]>
  // CHECK3: cir.get_member %[[#DEREF]][0] {name = "value"} : !cir.ptr<![[STRUCT]]> -> !cir.ptr<!s32i>
  arg->next->value;
}



//--- indirect_recursive_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/indirect_recursive_struct -o %t/indirect_recursive_struct.cir
// RUN: FileCheck --check-prefix=CHECK4 --input-file=%t/indirect_recursive_struct.cir %s

// Node B refers to A, and vice-versa, so a forward declaration is used to
// ensure the classes can be defined. Since types alias are not yet supported
// in recursive type, each struct is expanded until there are no more recursive
// types, or all the recursive types are self references.

// CHECK4: ![[B:.+]] = !cir.struct<struct "StructNodeB" {!cir.int<s, 32>, !cir.ptr<!cir.struct<struct "StructNodeA" {!cir.int<s, 32>, !cir.ptr<!cir.struct<struct "StructNodeB">>}
// CHECK4: ![[A:.+]] = !cir.struct<struct "StructNodeA" {!cir.int<s, 32>, !cir.ptr<!cir.struct<struct "StructNodeB" {!cir.int<s, 32>, !cir.ptr<!cir.struct<struct "StructNodeA">>}
struct StructNodeB;
struct StructNodeA {
  int value;
  struct StructNodeB *next;
};
struct StructNodeB {
  int value;
  struct StructNodeA *next;
};

void testIndirectSelfReference(struct StructNodeA arg) {
  // CHECK4: %[[#V1:]] = cir.get_member %{{.+}}[1] {name = "next"} : !cir.ptr<![[A]]> -> !cir.ptr<!cir.ptr<![[B]]>>
  // CHECK4: %[[#V2:]] = cir.load %[[#V1]] : cir.ptr <!cir.ptr<![[B]]>>, !cir.ptr<![[B]]>
  // CHECK4: %[[#V3:]] = cir.get_member %[[#V2]][1] {name = "next"} : !cir.ptr<![[B]]> -> !cir.ptr<!cir.ptr<![[A]]>>
  // CHECK4: %[[#V4:]] = cir.load %[[#V3]] : cir.ptr <!cir.ptr<![[A]]>>, !cir.ptr<![[A]]>
  // CHECK4: cir.get_member %[[#V4]][0] {name = "value"} : !cir.ptr<![[A]]> -> !cir.ptr<!s32i>
  arg.next->next->value;
}



//--- complex_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/complex_struct -o %t/complex_struct.cir
// RUN: FileCheck --check-prefix=CHECK5 --input-file=%t/complex_struct.cir %s

// A sizeable complex struct just to double check that stuff is working.

// CHECK5: !cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1">, !cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A">>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A">>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B">>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>>} #cir.record.decl.ast>
// CHECK5: !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C">, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A">>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B">>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>>, !cir.ptr<!cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C">, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B">>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>
// CHECK5: !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.struct<struct "anon.4">} #cir.record.decl.ast>} #cir.record.decl.ast>>} #cir.record.decl.ast>
// CHECK5: !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A">>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5">} #cir.record.decl.ast>} #cir.record.decl.ast>>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5">} #cir.record.decl.ast>>} #cir.record.decl.ast>} #cir.record.decl.ast>
// CHECK5: !cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B">} #cir.record.decl.ast>>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B">>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>
// CHECK5: !cir.struct<struct "A" {!cir.struct<struct "anon.1" {!cir.ptr<!cir.struct<struct "A">>} #cir.record.decl.ast>, !cir.struct<struct "B" {!cir.ptr<!cir.struct<struct "B">>, !cir.struct<struct "C" {!cir.ptr<!cir.struct<struct "A">>, !cir.ptr<!cir.struct<struct "B">>, !cir.ptr<!cir.struct<struct "C">>} #cir.record.decl.ast>, !cir.struct<union "anon.5" {!cir.ptr<!cir.struct<struct "A">>, !cir.struct<struct "anon.4" {!cir.ptr<!cir.struct<struct "B">>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>} #cir.record.decl.ast>
struct A {
  struct {
    struct A *a1;
  };
  struct B {
    struct B *b1;
    struct C {
      struct A *a2;
      struct B *b2;
      struct C *c1;
    } c;
    union {
      struct A *a2;
      struct {
        struct B *b3;
      };
    } u;
  } b;
};
void test(struct A *a){};
