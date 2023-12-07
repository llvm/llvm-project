// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

struct A {
    operator __auto_type() {} // expected-error {{'__auto_type' not allowed in conversion function type}}
};

__auto_type a() -> int; // expected-error {{function with trailing return type must specify return type 'auto'}}
__auto_type a2(); // expected-error {{'__auto_type' not allowed in function return type}}
template <typename T>
__auto_type b() { return T::x; } // expected-error {{'__auto_type' not allowed in function return type}}
auto c() -> __auto_type { __builtin_unreachable(); } // expected-error {{'__auto_type' not allowed in function return type}}
int d() {
  decltype(__auto_type) e = 1; // expected-error {{expected expression}}
  auto _ = [](__auto_type f) {}; // expected-error {{'__auto_type' not allowed in lambda parameter}}
  __auto_type g = 2;
  struct BitField { int field:2; };
  __auto_type h = BitField{1}.field; // (should work from C++)
  new __auto_type; // expected-error {{'__auto_type' not allowed in type allocated by 'new'}}
}

namespace TestDeductionFail {

template<typename T>
void caller(T x) {x.fun();} // expected-error {{parameter type 'TestDeductionFail::Abstract' is an abstract class}}

template<typename T>
auto getCaller(){
  return caller<T>; // expected-note {{in instantiation of function template specialization 'TestDeductionFail::caller<TestDeductionFail::Abstract>' requested here}}
}

class Abstract{
  public:
    void fun();
    virtual void vfun()=0; // expected-note {{unimplemented pure virtual method 'vfun' in 'Abstract'}}
    void call(){getCaller<Abstract>()(*this);} // expected-error {{allocating an object of abstract class type 'TestDeductionFail::Abstract'}}
};

}
