// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 std=c++11 %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid
// rdar://12378714

/**
 * \brief Aaa
*/
template<typename T> struct A {
/**
 * \brief Bbb
*/
  A();
/**
 * \brief Ccc
*/
  ~A();
/**
 * \brief Ddd
*/
  void f() { }
};
// CHECK: <Declaration>template &lt;typename T&gt; struct A {}</Declaration>
// CHECK: <Declaration>A&lt;T&gt;()</Declaration>
// CHECK: <Declaration>~A&lt;T&gt;()</Declaration>

/**
 * \Brief Eee
*/
template <typename T> struct D : A<T> {
/**
 * \brief
*/
  using A<T>::f;
  
  void f();
};
// CHECK: <Declaration>template &lt;typename T&gt; struct D :  A&lt;T&gt; {}</Declaration>
// CHECK: <Declaration>using A&lt;T&gt;::f</Declaration>

struct Base {
    int foo;
};
/**
 * \brief
*/
template<typename T> struct E : Base {
/**
 * \brief
*/
  using Base::foo;
};
// CHECK: <Declaration>template &lt;typename T&gt; struct E :  Base {}</Declaration>
// CHECK: <Declaration>using Base::foo</Declaration>

/// \tparam
/// \param AAA Blah blah
template<typename T>
void func_template_1(T AAA);
// CHECK: <Declaration>template &lt;typename T&gt; void func_template_1(T AAA)</Declaration>

template<template<template<typename CCC> class DDD, class BBB> class AAA>
void func_template_2();
// FIXME: There is not Declaration field in the generated output.

namespace rdar16128173 {
// CHECK: <Declaration>template &lt;class PtrTy&gt; class OpaquePtr {}</Declaration>

/// \brief Wrapper for void* pointer.
/// \tparam PtrTy Either a pointer type like 'T*' or a type that behaves like
///               a pointer.
template <class PtrTy>
class OpaquePtr {};

// CHECK: <Declaration>typedef OpaquePtr&lt;int&gt; DeclGroupPtrTy</Declaration>
typedef OpaquePtr<int> DeclGroupPtrTy;

DeclGroupPtrTy blah;
}
