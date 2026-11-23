// RUN: split-file %s %t


//--- incomplete_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/incomplete_struct -o %t/incomplete_struct.cir
// RUN: FileCheck %s --input-file=%t/incomplete_struct.cir --check-prefix=CHECK1

// Forward declaration of the record is never defined, so it is created as
// an incomplete struct in CIR and will remain as such.

// CHECK1: ![[INC_STRUCT:.+]] = !cir.struct<"IncompleteStruct" incomplete>
struct IncompleteStruct;
// CHECK1: testIncompleteStruct(%arg0: !cir.ptr<![[INC_STRUCT]]>
void testIncompleteStruct(struct IncompleteStruct *s) {};



//--- mutated_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/mutated_struct -o %t/mutated_struct.cir
// RUN: FileCheck %s --input-file=%t/mutated_struct.cir --check-prefix=CHECK2

// Foward declaration of the struct is followed by usage, then definition.
// This means it will initially be created as incomplete, then completed.

// CHECK2: ![[COMPLETE:.+]] = !cir.struct<"ForwardDeclaredStruct" {data !s32i}>
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

// CHECK3: ![[STRUCT:.+]] = !cir.struct<"RecursiveStruct" {data !s32i, data !cir.ptr<!cir.struct<"RecursiveStruct">>}>
struct RecursiveStruct {
  int value;
  struct RecursiveStruct *next;
};
// CHECK3: testRecursiveStruct(%arg0: !cir.ptr<![[STRUCT]]>
void testRecursiveStruct(struct RecursiveStruct *arg) {
  // CHECK3: %[[#NEXT:]] = cir.get_member %{{.+}}[1] {name = "next"} : !cir.ptr<![[STRUCT]]> -> !cir.ptr<!cir.ptr<![[STRUCT]]>>
  // CHECK3: %[[#DEREF:]] = cir.load{{.*}} %[[#NEXT]] : !cir.ptr<!cir.ptr<![[STRUCT]]>>, !cir.ptr<![[STRUCT]]>
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

// CHECK4: ![[B:.+]] = !cir.struct<"StructNodeB" {data !s32i, data !cir.ptr<!cir.struct<"StructNodeA" {data !s32i, data !cir.ptr<!cir.struct<"StructNodeB">>}
// CHECK4: ![[A:.+]] = !cir.struct<"StructNodeA" {data !s32i, data !cir.ptr<![[B]]>}>
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
  // CHECK4: %[[#V2:]] = cir.load{{.*}} %[[#V1]] : !cir.ptr<!cir.ptr<![[B]]>>, !cir.ptr<![[B]]>
  // CHECK4: %[[#V3:]] = cir.get_member %[[#V2]][1] {name = "next"} : !cir.ptr<![[B]]> -> !cir.ptr<!cir.ptr<![[A]]>>
  // CHECK4: %[[#V4:]] = cir.load{{.*}} %[[#V3]] : !cir.ptr<!cir.ptr<![[A]]>>, !cir.ptr<![[A]]>
  // CHECK4: cir.get_member %[[#V4]][0] {name = "value"} : !cir.ptr<![[A]]> -> !cir.ptr<!s32i>
  arg.next->next->value;
}



//--- complex_struct

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/complex_struct -o %t/complex_struct.cir
// RUN: FileCheck --check-prefix=CHECK5 --input-file=%t/complex_struct.cir %s

// A sizeable complex struct just to double check that stuff is working.
// CHECK5: !cir.struct<"anon.0" {data !cir.ptr<!cir.struct<"A" {data !cir.struct<"anon.0">, data !cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !cir.struct<"C" {data !cir.ptr<!cir.struct<"A">>, data !cir.ptr<!cir.struct<"B">>, data !cir.ptr<!cir.struct<"C">>}>, data !cir.union<"anon.1" {data !cir.ptr<!cir.struct<"A">>, data !cir.struct<"anon.2" {data !cir.ptr<!cir.struct<"B">>}>}>}>}>>}>
// CHECK5: !cir.struct<"C" {data !cir.ptr<!cir.struct<"A" {data !rec_anon2E0, data !cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !cir.struct<"C">, data !cir.union<"anon.1" {data !cir.ptr<!cir.struct<"A">>, data !cir.struct<"anon.2" {data !cir.ptr<!cir.struct<"B">>}>}>}>}>>, data !cir.ptr<!cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !cir.struct<"C">, data !cir.union<"anon.1" {data !cir.ptr<!cir.struct<"A" {data !rec_anon2E0, data !cir.struct<"B">}>>, data !cir.struct<"anon.2" {data !cir.ptr<!cir.struct<"B">>}>}>}>>, data !cir.ptr<!cir.struct<"C">>}>
// CHECK5: !cir.struct<"anon.2" {data !cir.ptr<!cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !rec_C, data !cir.union<"anon.1" {data !cir.ptr<!cir.struct<"A" {data !rec_anon2E0, data !cir.struct<"B">}>>, data !cir.struct<"anon.2">}>}>>}>
// CHECK5: !cir.union<"anon.1" {data !cir.ptr<!cir.struct<"A" {data !rec_anon2E0, data !cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !rec_C, data !cir.union<"anon.1">}>}>>, data !rec_anon2E2}>
// CHECK5: !cir.struct<"B" {data !cir.ptr<!cir.struct<"B">>, data !rec_C, data !rec_anon2E1}>
// CHECK5: !cir.struct<"A" {data !rec_anon2E0, data !rec_B}>
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


//--- incomplete_class_comma_expr
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir %t/incomplete_class_comma_expr -o %t/incomplete_class_comma_expr.cir
// RUN: FileCheck %s --input-file=%t/incomplete_class_comma_expr.cir --check-prefix=CHECK6
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-llvm %t/incomplete_class_comma_expr -o %t/incomplete_class_comma_expr-cir.ll
// RUN: FileCheck %s --input-file=%t/incomplete_class_comma_expr-cir.ll --check-prefix=CIR6
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -emit-llvm %t/incomplete_class_comma_expr -o %t/incomplete_class_comma_expr-ogcg.ll
// RUN: FileCheck %s --input-file=%t/incomplete_class_comma_expr-ogcg.ll --check-prefix=OGCG6

class Enum extern const writeTypeNames;
int flags = (writeTypeNames, flags);

// Incomplete type, never constant even though its an enum.
// CHECK6:      cir.global "private" external @writeTypeNames : !rec_Enum
// CHECK6-NOT:  constant

// CIR6:        @writeTypeNames = external global %class.Enum
// CIR6-NOT:    constant

// OGCG6:       @writeTypeNames = external global %class.Enum
// OGCG6-NOT:   constant

//--- extern_const_global_constant
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir %t/extern_const_global_constant -o %t/extern_const_global_constant.cir
// RUN: FileCheck %s --input-file=%t/extern_const_global_constant.cir --check-prefix=CHECK7
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-llvm %t/extern_const_global_constant -o %t/extern_const_global_constant-cir.ll
// RUN: FileCheck %s --input-file=%t/extern_const_global_constant-cir.ll --check-prefix=CIR7
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -emit-llvm %t/extern_const_global_constant -o %t/extern_const_global_constant-ogcg.ll
// RUN: FileCheck %s --input-file=%t/extern_const_global_constant-ogcg.ll --check-prefix=OGCG7

// Complete class with no mutable fields, can be marked constant.
struct NoMutable { int x; };
extern const NoMutable no_mutable_val;
// CHECK7:  cir.global "private" constant external @no_mutable_val : !rec_NoMutable
// Note: This is a case where CIR is doing a better job than classic, which
// always fails to exclude ctor/dtor.
// CIR7:    @no_mutable_val = external constant %struct.NoMutable
// OGCG7:   @no_mutable_val = external global %struct.NoMutable

// Complete class with a mutable field.
struct WithMutable { mutable int x; };
extern const WithMutable with_mutable_val;
// CHECK7:      cir.global "private" external @with_mutable_val : !rec_WithMutable
// CHECK7-NOT:  constant
// CIR7:        @with_mutable_val = external global %struct.WithMutable
// CIR7-NOT:    constant
// OGCG7:       @with_mutable_val = external global %struct.WithMutable

// Incomplete class - cannot be marked constant.
class Incomplete;
extern const Incomplete incomplete_val;
// CHECK7:      cir.global "private" external @incomplete_val : !rec_Incomplete
// CIR7:        @incomplete_val = external global %class.Incomplete
// OGCG7:       @incomplete_val = external global %class.Incomplete

void use(const NoMutable &, const WithMutable &, const Incomplete *);
void foo() { use(no_mutable_val, with_mutable_val, &incomplete_val); }
