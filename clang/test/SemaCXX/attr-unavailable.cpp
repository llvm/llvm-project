// RUN: %clang_cc1 -fsyntax-only -verify %s

int &foo(int); // expected-note {{candidate}}
double &foo(double); // expected-note {{candidate}}
void foo(...) __attribute__((__unavailable__)); // expected-note {{candidate function}} \
// expected-note{{'foo' has been explicitly marked unavailable here}}

void bar(...) __attribute__((__unavailable__)); // expected-note 2{{explicitly marked unavailable}} \
       // expected-note 2{{candidate function has been explicitly made unavailable}}

void test_foo(short* sp) {
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp); // expected-error{{call to unavailable function 'foo'}}

  void (*fp)(...) = &bar; // expected-error{{'bar' is unavailable}}
  void (*fp2)(...) = bar; // expected-error{{'bar' is unavailable}}

  int &(*fp3)(int) = foo;
  void (*fp4)(...) = foo; // expected-error{{'foo' is unavailable}}
}

namespace radar9046492 {
// rdar://9046492
#define FOO __attribute__((unavailable("not available - replaced")))

void foo() FOO; // expected-note {{candidate function has been explicitly made unavailable}}
void bar() {
  foo(); // expected-error {{call to unavailable function 'foo': not available - replaced}}
}
}

void unavail(short* sp)  __attribute__((__unavailable__));
void unavail(short* sp) {
  // No complains inside an unavailable function.
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp);
  foo();
}

// Show that delayed processing of 'unavailable' is the same
// delayed process for 'deprecated'.
// <rdar://problem/12241361> and <rdar://problem/15584219>
enum DeprecatedEnum { DE_A, DE_B } __attribute__((deprecated)); // expected-note {{'DeprecatedEnum' has been explicitly marked deprecated here}}
__attribute__((deprecated)) typedef enum DeprecatedEnum DeprecatedEnum;
typedef enum DeprecatedEnum AnotherDeprecatedEnum; // expected-warning {{'DeprecatedEnum' is deprecated}}

__attribute__((deprecated))
DeprecatedEnum testDeprecated(DeprecatedEnum X) { return X; }


enum UnavailableEnum { UE_A, UE_B } __attribute__((unavailable)); // expected-note {{'UnavailableEnum' has been explicitly marked unavailable here}}
__attribute__((unavailable)) typedef enum UnavailableEnum UnavailableEnum;
typedef enum UnavailableEnum AnotherUnavailableEnum; // expected-error {{'UnavailableEnum' is unavailable}}


__attribute__((unavailable))
UnavailableEnum testUnavailable(UnavailableEnum X) { return X; }


// Check that unavailable classes can be used as arguments to unavailable
// function, particularly in template functions.
#if !__has_feature(attribute_availability_in_templates)
#error "Missing __has_feature"
#endif
class __attribute((unavailable)) UnavailableClass; // \
        expected-note 3{{'UnavailableClass' has been explicitly marked unavailable here}}
void unavail_class(UnavailableClass&); // expected-error {{'UnavailableClass' is unavailable}}
void unavail_class_marked(UnavailableClass&) __attribute__((unavailable));
template <class T> void unavail_class(UnavailableClass&); // expected-error {{'UnavailableClass' is unavailable}}
template <class T> void unavail_class_marked(UnavailableClass&) __attribute__((unavailable));
template <class T> void templated(T&);
void untemplated(UnavailableClass &UC) {  // expected-error {{'UnavailableClass' is unavailable}}
  templated(UC);
}
void untemplated_marked(UnavailableClass &UC) __attribute__((unavailable)) {
  templated(UC);
}

template <class T> void templated_calls_bar() { bar(); } // \
           // expected-error{{call to unavailable function 'bar'}}
template <class T> void templated_calls_bar_arg(T v) { bar(v); } // \
           // expected-error{{call to unavailable function 'bar'}}
template <class T> void templated_calls_bar_arg_never_called(T v) { bar(v); }

template <class T>
void unavail_templated_calls_bar() __attribute__((unavailable)) { // \
  expected-note{{candidate function [with T = int] has been explicitly made unavailable}}
  bar(5);
}
template <class T>
void unavail_templated_calls_bar_arg(T v) __attribute__((unavailable)) { // \
  expected-note{{candidate function [with T = int] has been explicitly made unavailable}}
  bar(v);
}

void calls_templates_which_call_bar() {
  templated_calls_bar<int>();

  templated_calls_bar_arg(5); // \
  expected-note{{in instantiation of function template specialization 'templated_calls_bar_arg<int>' requested here}}

  unavail_templated_calls_bar<int>(); // \
  expected-error{{call to unavailable function 'unavail_templated_calls_bar'}}

  unavail_templated_calls_bar_arg(5); // \
  expected-error{{call to unavailable function 'unavail_templated_calls_bar_arg'}}
}

template <class T> void unavail_templated(T) __attribute__((unavailable)); // \
           expected-note{{candidate function [with T = int] has been explicitly made unavailable}}
void calls_unavail_templated() {
  unavail_templated(5); // expected-error{{call to unavailable function 'unavail_templated'}}
}
void unavail_calls_unavail_templated() __attribute__((unavailable)) {
  unavail_templated(5);
}

void unavailable() __attribute((unavailable)); // \
       expected-note 4{{candidate function has been explicitly made unavailable}}
struct AvailableStruct {
  void calls_unavailable() { unavailable(); } // \
  expected-error{{call to unavailable function 'unavailable'}}
  template <class U> void calls_unavailable() { unavailable(); } // \
  expected-error{{call to unavailable function 'unavailable'}}
};
template <class T> struct AvailableStructTemplated {
  void calls_unavailable() { unavailable(); } // \
  expected-error{{call to unavailable function 'unavailable'}}
  template <class U> void calls_unavailable() { unavailable(); } // \
  expected-error{{call to unavailable function 'unavailable'}}
};
struct __attribute__((unavailable)) UnavailableStruct {
  void calls_unavailable() { unavailable(); }
  template <class U> void calls_unavailable() { unavailable(); }
};
template <class T> struct __attribute__((unavailable)) UnavailableStructTemplated {
  void calls_unavailable() { unavailable(); }
  template <class U> void calls_unavailable() { unavailable(); }
};
