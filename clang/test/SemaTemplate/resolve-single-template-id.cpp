// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace std {
  class type_info {};
}

void one() { }
void two() { } // expected-note 4{{possible target for call}}
void two(int) { } // expected-note 4{{possible target for call}}

template<class T> void twoT() { } // expected-note 5{{possible target for call}}
template<class T> void twoT(int) { } // expected-note 5{{possible target for call}}

template<class T> void oneT() { }
template<class T, class U> void oneT(U) { }
/*
The target can be
 an object or reference being initialized (8.5, 8.5.3),
 the left side of an assignment (5.17),
 a parameter of a function (5.2.2),
 a parameter of a user-defined operator (13.5),
 the return value of a function, operator function, or conversion (6.6.3),
 an explicit type conversion (5.2.3, 5.2.9, 5.4), or
 a non-type template-parameter (14.3.2)
*/
//#include <typeinfo>
template<void (*p)(int)> struct test { };

int main()
{
   one;         // expected-warning {{expression result unused}}
   two;         // expected-error {{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
   oneT<int>;  // expected-warning {{expression result unused}}
   twoT<int>;  // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
   typeid(oneT<int>); // expected-warning{{expression result unused}}
  sizeof(oneT<int>); // expected-error {{invalid application of 'sizeof' to a function type}}
  sizeof(twoT<int>); //expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
  decltype(oneT<int>)* fun = 0;
  
  *one;    // expected-warning {{expression result unused}}
  *oneT<int>;   // expected-warning {{expression result unused}}
  *two;  //expected-error {{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} expected-error {{indirection requires pointer operand}}
  *twoT<int>; //expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
  !oneT<int>;  // expected-warning {{expression result unused}} expected-warning {{address of function 'oneT<int>' will always evaluate to 'true'}} expected-note {{prefix with the address-of operator to silence this warning}}
  +oneT<int>;  // expected-warning {{expression result unused}}
  -oneT<int>;  //expected-error {{invalid argument type}}
  oneT<int> == 0;   // expected-warning {{equality comparison result unused}} \
                    // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                    // expected-warning {{comparison of function 'oneT<int>' equal to a null pointer is always false}} \
                    // expected-note {{prefix with the address-of operator to silence this warning}}
  0 == oneT<int>;   // expected-warning {{equality comparison result unused}} \
                    // expected-warning {{comparison of function 'oneT<int>' equal to a null pointer is always false}} \
                    // expected-note {{prefix with the address-of operator to silence this warning}}
  0 != oneT<int>;   // expected-warning {{inequality comparison result unused}} \
                    // expected-warning {{comparison of function 'oneT<int>' not equal to a null pointer is always true}} \
                    // expected-note {{prefix with the address-of operator to silence this warning}}
  (false ? one : oneT<int>);   // expected-warning {{expression result unused}}
  void (*p1)(int); p1 = oneT<int>;
  
  int i = (int) (false ? (void (*)(int))twoT<int> : oneT<int>); //expected-error {{incompatible operand}}
  (twoT<int>) == oneT<int>; //expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}} {{cannot resolve overloaded function 'twoT' from context}}
  bool b = oneT<int>; // expected-warning {{address of function 'oneT<int>' will always evaluate to 'true'}} expected-note {{prefix with the address-of operator to silence this warning}}
  void (*p)() = oneT<int>;
  test<oneT<int> > ti;
  void (*u)(int) = oneT<int>;

  b = (void (*)()) twoT<int>;

  one < one; // expected-warning {{self-comparison always evaluates to false}} \
             // expected-warning {{relational comparison result unused}}       \
             // expected-warning {{ordered comparison of function pointers}}

  oneT<int> < oneT<int>; // expected-warning {{self-comparison always evaluates to false}} \
                         // expected-warning {{relational comparison result unused}}       \
                         // expected-warning {{ordered comparison of function pointers}}

  two < two; //expected-error 2 {{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} expected-error {{invalid operands to binary expression ('void' and 'void')}}
  twoT<int> < twoT<int>; //expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}} {{cannot resolve overloaded function 'twoT' from context}}
  oneT<int> == 0;   // expected-warning {{equality comparison result unused}} \
                    // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                    // expected-warning {{comparison of function 'oneT<int>' equal to a null pointer is always false}} \
                    // expected-note {{prefix with the address-of operator to silence this warning}}

}

struct rdar9108698 {
  template<typename> void f(); // expected-note{{possible target for call}}
};

void test_rdar9108698(rdar9108698 x) {
  x.f<int>; // expected-error{{reference to non-static member function must be called}}
}

namespace GCC_PR67898 {
  void f(int);
  void f(float);
  template<typename T, T F, T G, bool b = F == G> struct X {
    static_assert(b, "");
  };
  template<typename T> void test1() { X<void(T), f, f>(); }
  template<typename T> void test2() { X<void(*)(T), f, f>(); }
  template void test1<int>();
  template void test2<int>();
}
