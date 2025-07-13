 // RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
 // RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
 
class D {
    class E{
        class F{}; // expected-note{{implicitly declared private here}}
        friend  void foo(D::E::F& q);
        };
    friend  void foo(D::E::F& q); // expected-error{{'F' is a private member of 'D::E'}}
    };

void foo(D::E::F& q) {}

class D1 {
    class E1{
        class F1{}; // expected-note{{implicitly declared private here}}
        friend  D1::E1::F1 foo1();
        };
    friend  D1::E1::F1 foo1(); // expected-error{{'F1' is a private member of 'D1::E1'}}
    };

D1::E1::F1 foo1() { return D1::E1::F1(); }

class D2 {
    class E2{
        class F2{};
        friend  void foo2();
        };
    friend  void foo2(){ D2::E2::F2 c;}
    };
