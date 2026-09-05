// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// List-initializing a temporary array of unknown bound from an empty list
// creates a zero-length array whose element type was never completed, and
// destructor lookup on the uninstantiated (or forward-declared) element type
// crashed. https://github.com/llvm/llvm-project/issues/217883

namespace gh217883 {
template <typename> struct Q {};

const Q<int> (&r1)[] = {};
Q<int> (&&r2)[] = {};
const Q<int> (&r3)[][2] = {};

void call(void (*f)(const Q<int> (&)[])) { f({}); }
void call_rvalue(void (*f)(Q<int> (&&)[])) { f({}); }
void call_nested(void (*f)(const Q<int> (&)[][2])) { f({}); }

#if __cplusplus >= 202002L
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
