// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.RefCntblBaseVirtualDtor -std=c++17 -verify %s
// expected-no-diagnostics

template<class T> struct Alloc {};
template<class T, class A> struct Storage {};
template<class T, class A=Alloc<T>> struct Node : Storage<T,A> {};
struct Derived : Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<int>>>>>>>>>>>>>>>>>>>>>>>>>>>> {};
