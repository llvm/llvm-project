// RUN: %clang_cc1 -std=c++14 -triple nvptx64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple nvptx64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple nvptx64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct some_struct {
    int x;
};

template<int I>
int var_template;

template<> int var_template<0>;
template<> int var_template<1> = 1;
template<> some_struct var_template<2>;

// CIR: !rec_some_struct = !cir.record<struct "some_struct" {!s32i}>
// CIR: cir.global external @_Z12var_templateILi0EE = #cir.int<0> : !s32i
// CIR: cir.global external @_Z12var_templateILi1EE = #cir.int<1> : !s32i
// CIR: cir.global external @_Z12var_templateILi2EE = #cir.zero : !rec_some_struct

// LLVM: %[[STRUCT_TYPE:.+]] = type { i32 }
// LLVM: @_Z12var_templateILi0EE = global i32 0
// LLVM: @_Z12var_templateILi1EE = global i32 1
// LLVM: @_Z12var_templateILi2EE = global %[[STRUCT_TYPE]] zeroinitializer

// OGCG: %[[STRUCT_TYPE:.+]] = type { i32 }
// OGCG: @_Z12var_templateILi0EE = global i32 0
// OGCG: @_Z12var_templateILi1EE = global i32 1
// OGCG: @_Z12var_templateILi2EE = global %[[STRUCT_TYPE]] zeroinitializer

template<typename T, int I> int partial_var_template_specialization_shouldnt_hit_codegen;
template<typename T> int partial_var_template_specialization_shouldnt_hit_codegen<T, 123>;
template<int I> float partial_var_template_specialization_shouldnt_hit_codegen<float, I>;

// CIR-NOT: partial_var_template_specialization_shouldnt_hit_codegen
// LLVM-NOT: partial_var_template_specialization_shouldnt_hit_codegen
// OGCG-NOT: partial_var_template_specialization_shouldnt_hit_codegen
