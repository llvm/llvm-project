// RUN: %clang_cc1 -std=c++2a -fconcepts-ts -x c++ -verify %s

template<typename T>
constexpr bool is_ptr_v = false;

template<typename T>
constexpr bool is_ptr_v<T*> = true;

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T> requires is_ptr_v<T> // expected-note   {{because 'is_ptr_v<int>' evaluated to false}}
                         // expected-note@-1{{because 'is_ptr_v<char>' evaluated to false}}
auto dereference(T t) { // expected-note   {{candidate template ignored: constraints not satisfied [with T = int]}}
                        // expected-note@-1{{candidate template ignored: constraints not satisfied [with T = char]}}
  return *t;
}

static_assert(is_same_v<decltype(dereference<int*>(nullptr)), int>);
static_assert(is_same_v<decltype(dereference(2)), int>); // expected-error {{no matching function for call to 'dereference'}}
static_assert(is_same_v<decltype(dereference<char>('a')), char>); // expected-error {{no matching function for call to 'dereference'}}


template<typename T> requires T{} + T{} // expected-note {{because substituted constraint expression is ill-formed: invalid operands to binary expression ('A' and 'A')}}
auto foo(T t) { // expected-note {{candidate template ignored: constraints not satisfied [with T = A]}}
  return t + t;
}


template<typename T> requires !((T{} - T{}) && (T{} + T{})) || false
// expected-note@-1{{because substituted constraint expression is ill-formed: invalid operands to binary expression ('A' and 'A')}}
// expected-note@-2{{and 'false' evaluated to false}}
auto bar(T t) { // expected-note {{candidate template ignored: constraints not satisfied [with T = A]}}
  return t + t;
}

struct A { };

static_assert(foo(A{})); // expected-error {{no matching function for call to 'foo'}}
static_assert(bar(A{})); // expected-error {{no matching function for call to 'bar'}}