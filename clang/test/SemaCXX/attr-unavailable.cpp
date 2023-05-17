// RUN: %clang_cc1 -fsyntax-only -verify %s

int &foo(int);
double &foo(double);
void foo(...) __attribute__((__unavailable__)); // \
// expected-note 2 {{'foo' has been explicitly marked unavailable here}}

void bar(...) __attribute__((__unavailable__)); // expected-note 4 {{explicitly marked unavailable}}

void test_foo(short* sp) {
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp); // expected-error{{'foo' is unavailable}}

  void (*fp)(...) = &bar; // expected-error{{'bar' is unavailable}}
  void (*fp2)(...) = bar; // expected-error{{'bar' is unavailable}}

  int &(*fp3)(int) = foo;
  void (*fp4)(...) = foo; // expected-error{{'foo' is unavailable}}
}

namespace radar9046492 {
// rdar://9046492
#define FOO __attribute__((unavailable("not available - replaced")))

void foo() FOO; // expected-note{{'foo' has been explicitly marked unavailable here}}
void bar() {
  foo(); // expected-error {{'foo' is unavailable: not available - replaced}}
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
typedef enum DeprecatedEnum AnotherDeprecatedEnum; // expected-warning {{'DeprecatedEnum' is deprecated}}

__attribute__((deprecated)) typedef enum DeprecatedEnum DeprecatedEnum;
__attribute__((deprecated))
DeprecatedEnum testDeprecated(DeprecatedEnum X) { return X; }


enum UnavailableEnum { UE_A, UE_B } __attribute__((unavailable)); // expected-note {{'UnavailableEnum' has been explicitly marked unavailable here}}
typedef enum UnavailableEnum AnotherUnavailableEnum; // expected-error {{'UnavailableEnum' is unavailable}}
                                                     //

__attribute__((unavailable)) typedef enum UnavailableEnum UnavailableEnum;
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
           // expected-error{{'bar' is unavailable}}
template <class T> void templated_calls_bar_arg(T v) { bar(v); } // \
           // expected-error{{'bar' is unavailable}}
template <class T> void templated_calls_bar_arg_never_called(T v) { bar(v); }

template <class T>
void unavail_templated_calls_bar() __attribute__((unavailable)) { //  \
// expected-note {{'unavail_templated_calls_bar<int>' has been explicitly marked unavailable here}}
  bar(5);
}
template <class T>
void unavail_templated_calls_bar_arg(T v) __attribute__((unavailable)) {
// expected-note@-1 {{'unavail_templated_calls_bar_arg<int>' has been explicitly marked unavailable here}}
  bar(v);
}

void calls_templates_which_call_bar() {
  templated_calls_bar<int>();

  templated_calls_bar_arg(5); // \
  expected-note{{in instantiation of function template specialization 'templated_calls_bar_arg<int>' requested here}}

  unavail_templated_calls_bar<int>(); // \
  expected-error{{'unavail_templated_calls_bar<int>' is unavailable}}

  unavail_templated_calls_bar_arg(5); // \
  expected-error{{'unavail_templated_calls_bar_arg<int>' is unavailable}}
}

template <class T> void unavail_templated(T) __attribute__((unavailable));
// expected-note@-1 {{'unavail_templated<int>' has been explicitly marked unavailable here}}
void calls_unavail_templated() {
  unavail_templated(5); // expected-error{{'unavail_templated<int>' is unavailable}}
}
void unavail_calls_unavail_templated() __attribute__((unavailable)) {
  unavail_templated(5);
}

void unavailable() __attribute((unavailable));
// expected-note@-1 4 {{'unavailable' has been explicitly marked unavailable here}}
struct AvailableStruct {
  void calls_unavailable() { unavailable(); } // \
  expected-error{{'unavailable' is unavailable}}
  template <class U> void calls_unavailable() { unavailable(); } // \
  expected-error{{'unavailable' is unavailable}}
};
template <class T> struct AvailableStructTemplated {
  void calls_unavailable() { unavailable(); } // \
  expected-error{{'unavailable' is unavailable}}
  template <class U> void calls_unavailable() { unavailable(); } // \
  expected-error{{'unavailable' is unavailable}}
};
struct __attribute__((unavailable)) UnavailableStruct {
  void calls_unavailable() { unavailable(); }
  template <class U> void calls_unavailable() { unavailable(); }
};
template <class T> struct __attribute__((unavailable)) UnavailableStructTemplated {
  void calls_unavailable() { unavailable(); }
  template <class U> void calls_unavailable() { unavailable(); }
};

int unavailable_int() __attribute__((unavailable)); // expected-note 2 {{'unavailable_int' has been explicitly marked unavailable here}}
int has_default_arg(int x = unavailable_int()) { // expected-error{{'unavailable_int' is unavailable}}
  return x;
}

int has_default_arg2(int x = unavailable_int()) __attribute__((unavailable)) {
  return x;
}

template <class T>
T unavailable_template() __attribute__((unavailable));
// expected-note@-1 {{'unavailable_template<int>' has been explicitly marked unavailable here}}

template <class T>
int has_default_arg_template(T x = unavailable_template<T>()) {}
// expected-error@-1 {{'unavailable_template<int>' is unavailable}}

int instantiate_it = has_default_arg_template<int>();
// expected-note@-1 {{in instantiation of default function argument expression for 'has_default_arg_template<int>' required here}}

template <class T>
int has_default_arg_template2(T x = unavailable_template<T>())
    __attribute__((unavailable)) {}

__attribute__((unavailable))
int instantiate_it2 = has_default_arg_template2<int>();

template <class T>
int phase_one_unavailable(int x = unavailable_int()) {}
// expected-error@-1 {{'unavailable_int' is unavailable}}

template <class T>
int phase_one_unavailable2(int x = unavailable_int()) __attribute__((unavailable)) {}

namespace GH61815 {
template <class _ValueType = int>
class __attribute__((unavailable)) polymorphic_allocator {}; // expected-note 2 {{'polymorphic_allocator<void>' has been explicitly marked unavailable here}}

void f() {
  polymorphic_allocator<void> a; // expected-error {{'polymorphic_allocator<void>' is unavailable}}
  polymorphic_allocator<void> b; // expected-error {{'polymorphic_allocator<void>' is unavailable}}
}
}
