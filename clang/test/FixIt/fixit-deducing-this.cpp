// RUN: %clang_cc1 -verify -std=c++2c %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++2c -fixit %t
// RUN: %clang_cc1 -x c++ -std=c++2c %t
// RUN: not %clang_cc1 -std=c++2c -x c++ -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
struct S {
    void f(this S);
    void g() {
        (void)&f;  // expected-error {{must explicitly qualify name of member function when taking its address}}
// CHECK: fix-it:{{.*}}:{9:16-9:16}:"S::"
    }
};

struct S2 {
    void f(this S2 foo) {
        g(); // expected-error {{call to non-static member function without an object argument}}
// CHECK: fix-it:{{.*}}:{16:9-16:9}:"foo."

        h(); // expected-error {{call to explicit member function without an object argument}}
// CHECK: fix-it:{{.*}}:{19:9-19:9}:"foo."

        i();

        var; // expected-error {{invalid use of member 'var' in explicit object member function}}
// CHECK: fix-it:{{.*}}:{24:9-24:9}:"foo."

    }
    void g();
    void h(this S2 s);
    static void i();
    int var;
};
