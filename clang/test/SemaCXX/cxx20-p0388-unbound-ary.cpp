// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s

// p0388 conversions to unbounded array
// dcl.init.list/3

namespace One {
int ga[1];

auto &frob1() {
  int(&r1)[] = ga;
#if __cplusplus < 202002
  // expected-error@-2{{cannot bind to a value of unrelated type}}
#endif

  return r1;
}

auto &frob2(int (&arp)[1]) {
  int(&r2)[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{cannot bind to a value of unrelated type}}
#endif

  return r2;
}
} // namespace One

namespace Two {
int ga[1];

auto *frob1() {
  int(*r1)[] = &ga;
#if __cplusplus < 202002
  // expected-error@-2{{with an rvalue of type}}
#endif

  return r1;
}

auto *frob2(int (*arp)[1]) {
  int(*r2)[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{with an lvalue of type}}
#endif

  return r2;
}
} // namespace Two

namespace Four {
using Inc = int[2];
using Mat = Inc[1];
Mat *ga[2];

auto *frob1() {
  Inc(*const(*r1)[])[] = &ga;
#if __cplusplus < 202002
  // expected-error@-2{{with an rvalue of type}}
#else
  // missing a required 'const'
  Inc(*(*r2)[])[] = &ga; // expected-error{{cannot initialize}}
#endif

  return r1;
}

auto *frob2(Mat *(*arp)[1]) {
  Inc(*const(*r2)[])[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{with an lvalue of type}}
#else
  Inc(*(*r3)[])[] = arp; // expected-error{{cannot initialize}}
#endif

  return r2;
}

} // namespace Four

namespace Five {
// from the paper
char (&b(int(&&)[]))[1];   // #1
char (&b(long(&&)[]))[2];  // #2
char (&b(int(&&)[1]))[3];  // #3
char (&b(long(&&)[1]))[4]; // #4
char (&b(int(&&)[2]))[5];  // #5
#if __cplusplus < 202002
    // expected-note@-6{{cannot convert initializer}}
    // expected-note@-6{{cannot convert initializer}}
    // expected-note@-6{{too many initializers}}
    // expected-note@-6{{too many initializers}}
    // expected-note@-6{{too many initializers}}
#endif

void f() {
  static_assert(sizeof(b({1})) == 3);
  static_assert(sizeof(b({1, 2})) == 5);
  static_assert(sizeof(b({1, 2, 3})) == 1);
#if __cplusplus < 202002
  // expected-error@-2{{no matching function}}
#endif
}
} // namespace Five

#if __cplusplus >= 202002
namespace Six {
// from over.ics.rank 3.1
char (&f(int(&&)[]))[1];    // #1
char (&f(double(&&)[]))[2]; // #2
char (&f(int(&&)[2]))[3];   // #3

void toto() {
  // Calls #1: Better than #2 due to conversion, better than #3 due to bounds
  static_assert(sizeof(f({1})) == 1);

  // Calls #2: Identity conversion is better than floating-integral conversion
  static_assert(sizeof(f({1.0})) == 2);

  // Calls #2: Identity conversion is better than floating-integral conversion
  static_assert(sizeof(f({1.0, 2.0})) == 2);

  // Calls #3: Converting to array of known bound is better than to unknown
  //           bound, and an identity conversion is better than
  //           floating-integral conversion
  static_assert(sizeof(f({1, 2})) == 3);
}

} // namespace Six

namespace Seven {

char (&f(int(&&)[]))[1];     // #1
char (&f(double(&&)[1]))[2]; // #2

void quux() {
  // Calls #2, float-integral conversion rather than create zero-sized array
  static_assert(sizeof(f({})) == 2);
}

} // namespace Seven

namespace Eight {

// brace-elision is not a thing here:
struct A {
  int x, y;
};

char (&f1(int(&&)[]))[1]; // #1
char (&f1(A(&&)[]))[2];   // #2

void g1() {
  // pick #1, even though that is more elements than #2
  // 6 ints, as opposed to 3 As
  static_assert(sizeof(f1({1, 2, 3, 4, 5, 6})) == 1);
}

void f2(A(&&)[]); // expected-note{{candidate function not viable}}
void g2() {
  f2({1, 2, 3, 4, 5, 6}); // expected-error{{no matching function}}
}

void f3(A(&&)[]);
void g3() {
  auto &f = f3;

  f({1, 2, 3, 4, 5, 6}); // OK! We're coercing to an already-selected function
}

} // namespace Eight

#endif

namespace gh217883 {
template <typename> struct Q {};

const Q<int> (&r1)[] = {};
Q<int> (&&r2)[] = {};
const Q<int> (&r3)[][2] = {};

void call(void (*f)(const Q<int> (&)[])) { f({}); }
void call_rvalue(void (*f)(Q<int> (&&)[])) { f({}); }
void call_nested(void (*f)(const Q<int> (&)[][2])) { f({}); }

#if __cplusplus >= 202002
static_assert(requires(void f(const Q<int> (&)[])) { f({}); });
#endif

template <typename> struct DeletedDtor { ~DeletedDtor() = delete; }; // expected-note {{marked deleted here}}
void call_deleted(void (*f)(const DeletedDtor<int> (&)[])) {
  f({}); // expected-error {{attempt to use a deleted function}}
}

struct Incomplete; // expected-note 3 {{forward declaration of 'gh217883::Incomplete'}}
const Incomplete (&r4)[] = {}; // expected-error {{initialization of incomplete type 'const Incomplete'}}
const Incomplete (&r5)[][2] = {}; // expected-error {{initialization of incomplete type 'const Incomplete'}}
void call_incomplete(void (*f)(const Incomplete (&)[])) {
  f({}); // expected-error {{initialization of incomplete type 'const Incomplete'}}
}
} // namespace gh217883
