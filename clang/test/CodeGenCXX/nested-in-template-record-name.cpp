// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s --check-prefix=CIR %}

// The template arguments of an enclosing class template specialization are not
// part of the name given to a nested type in the IR: they can be arbitrarily
// large and add nothing, since the IR makes the names unique anyway.

template <class T> struct Outer {
  struct Inner { T x; };
  template <class U> struct Member { U y; };
};

Outer<int>::Inner a;
Outer<char>::Inner b;
Outer<int>::Member<float> c;

// This also applies to a nested type used as a template argument of the type
// being named.
template <class T> struct Wrap { T x; };
Wrap<Outer<int>::Inner> d;
Wrap<Outer<char>::Inner> e;

// LLVM-DAG: %"struct.Outer::Inner" = type { i32 }
// LLVM-DAG: %"struct.Outer::Inner.{{[0-9]+}}" = type { i8 }
// LLVM-DAG: %"struct.Outer::Member" = type { float }

// CIR keeps the arguments of the type itself, but they, like the enclosing
// types', don't include those of any enclosing specialization.
// CIR-DAG: !cir.struct<"Outer::Inner" {data !s32i}>
// CIR-DAG: !cir.struct<"Outer::Inner.{{[0-9]+}}" {data !s8i}>
// CIR-DAG: !cir.struct<"Outer::Member<float>" {data !cir.float}>
// CIR-DAG: !cir.struct<"Wrap<Outer::Inner>" {data !rec_Outer3A3AInner}>
// CIR-DAG: !cir.struct<"Wrap<Outer::Inner>.{{[0-9]+}}" {data !rec_Outer3A3AInner2E{{[0-9]+}}}>
