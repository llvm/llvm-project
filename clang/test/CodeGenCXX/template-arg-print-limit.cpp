// RUN: %clang_cc1 -std=c++11 -ftemplate-depth=100 -triple x86_64-unknown-linux-gnu -emit-llvm -ftype-print-limit=5 %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -fclangir -std=c++11 -ftemplate-depth=100 -triple x86_64-unknown-linux-gnu -emit-cir -ftype-print-limit=5 %s -o - | FileCheck %s --check-prefix=CIR %}

// Types whose full spelling is astronomically large must not take forever to
// print when giving them a name in the IR.

template <class A, class B> struct F {
  struct Inner { int x; };
};

// Each level uses the previous one twice: this is 2^40 'int's when spelled out.
template <int N> struct Chain {
  using type = F<typename Chain<N - 1>::type, typename Chain<N - 1>::type>;
};
template <> struct Chain<0> { using type = int; };

F<int, char> small;
Chain<40>::type huge;
Chain<40>::type::Inner hugeInner;

// Classic codegen names a class template specialization without its template
// arguments, but not a type nested inside it, which has the enclosing type's.
// LLVM-DAG: %struct.F = type { i8 }
// LLVM-DAG: %struct.F.0 = type { i8 }
// LLVM-DAG: %"struct.F<F<F<F<F<F<[2 args omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>::Inner" = type { i32 }

// CIR spells out the template arguments in the name of the type.
// CIR-DAG: !cir.struct<"F<int, char>" {pad !u8i}>
// CIR-DAG: !cir.struct<"F<F<F<F<F<F<[2 args omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>" {pad !u8i}>
// CIR-DAG: !cir.struct<"F<F<F<F<F<F<[2 args omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>, [arg omitted]>::Inner" {data !s32i}>
