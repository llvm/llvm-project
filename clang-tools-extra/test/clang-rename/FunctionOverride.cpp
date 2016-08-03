// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=318 -new-name=bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class A { virtual void foo(); };    // CHECK: class A { virtual void bar(); };
class B : public A { void foo(); }; // CHECK: class B : public A { void bar(); };
class C : public B { void foo(); }; // CHECK: class C : public B { void bar(); };

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
