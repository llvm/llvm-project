// RUN: %clang_cc1 -std=c++20 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fblocks -fsyntax-only -verify %s

namespace aliasing_decltype {
template <class T> void foo(T a, decltype(a)); // #foo1
template <class T> void foo(T a, decltype(a), int = 0); // #foo2
template <class T> void bar(T a, decltype(a)); // #bar1
template <class T> void bar(T a, decltype(a), int = 0); // #bar2
void trigger() {
  foo(0, 0);
  // expected-error@-1 {{call to 'foo' is ambiguous}}
  // expected-note@#foo1 {{candidate function}}
  // expected-note@#foo2 {{candidate function}}
  bar(0, 0);
  // expected-error@-1 {{call to 'bar' is ambiguous}}
  // expected-note@#bar1 {{candidate function}}
  // expected-note@#bar2 {{candidate function}}
}
}

namespace ambiguous_overload {

template <class> struct S {
  template <class T> S(T);
};

struct S1 {};
struct S2 {
  operator S1();
};

template <typename T> auto foo(T, S<decltype(0)>); // #ambiguous_overload1
template <typename T> auto foo(T arg, decltype(arg)) { foo(arg, S2{}); }
// expected-error@-1 {{function 'foo<ambiguous_overload::S1>' with deduced return type cannot be used before it is defined}}
void bar(S1 d) { foo(d, S1{}); }
// expected-note@-1 {{in instantiation of function template specialization 'ambiguous_overload::foo<ambiguous_overload::S1>' requested here}}
// expected-note@#ambiguous_overload1 {{'foo<ambiguous_overload::S1>' declared here}}
}

namespace explicit_specialization {
template <typename T> void foo(T, int);
template <typename T> void foo(T arg, decltype(arg));
template <> void foo(int, int) {}
}

namespace recursive_lambda {
template <typename Func>
auto foo(Func func, decltype(func()) (*bar)()) -> decltype(func()) { return bar(); }
template <typename Func>
auto foo(Func func, decltype(func()) Value) -> decltype(func()) {
  return foo(func, [=] { return Value; });
}
void *foo(void *(*func)()) { return foo(func, nullptr); }
}

namespace dependent_nested_param {
struct X { using Nested = int; };
template <class T> void foo(typename T::Nested a, decltype(a)); // #dependent_nested_param1
template <class T> void foo(typename T::Nested a, decltype(a), int = 0); // #dependent_nested_param2
void trigger() { foo<X>(0, 0); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#dependent_nested_param1 {{candidate function}}
// expected-note@#dependent_nested_param2 {{candidate function}}
}

namespace nested_decltype_type {
struct X { using Nested = int; };
template <class T> void foo(T a, typename decltype(a)::Nested); // #nested_decltype_type1
template <class T> void foo(T a, typename decltype(a)::Nested, int = 0); // #nested_decltype_type2
void trigger() { foo<X>(X{}, 0); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#nested_decltype_type1 {{candidate function}}
// expected-note@#nested_decltype_type2 {{candidate function}}
}

namespace auto_decltype {
void foo(auto a, decltype(a)); // #auto_decltype1
void foo(auto a, decltype(a), int = 0); // #auto_decltype2
void trigger() { foo(0, 0); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#auto_decltype1 {{candidate function}}
// expected-note@#auto_decltype2 {{candidate function}}
}

namespace wrapped_decltype {
template <class> struct S {};
template <class T> void foo(T a, S<decltype(a)>); // #wrapped_decltype1
template <class T> void foo(T a, S<decltype(a)>, int = 0); // #wrapped_decltype2
void trigger() { foo(0, S<int>{}); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#wrapped_decltype1 {{candidate function}}
// expected-note@#wrapped_decltype2 {{candidate function}}
}

namespace variadic_decltype {
template <class T, class... Ts> void foo(T a, decltype(a), Ts...);
template <class T> void foo(T a, decltype(a));
void bar() { foo(0, 0); }
}

namespace pack_decltype {
void foo(auto a, decltype(a), auto...);
void foo(auto a, decltype(a));
void trigger() { foo(0, 0); }
}

namespace candidate_deleted {
template <class T> void foo(T a, decltype(a)); // #non_deleted
template <class T> void foo(T *a, decltype(a)) = delete; // #deleted_candidate
void trigger() { int x; foo(&x, &x); }
// expected-error@-1 {{call to deleted function 'foo'}}
// expected-note@#non_deleted {{candidate function}}
// expected-note@#deleted_candidate {{candidate function}}
}

namespace default_decltype {
template <class T> void foo(T a, int, decltype(a) = 0); // #default_decltype1
template <class T> void foo(T a, int, decltype(a) = 0, int = 0); // #default_decltype2
void trigger() { foo(0, 0); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#default_decltype1 {{candidate function}}
// expected-note@#default_decltype2 {{candidate function}}
}

namespace default_lambda {
template <class T> void foo(T a, auto f = [](decltype(a)){}); // #default_lambda1
template <class T> void foo(T a, auto f = [](decltype(a)){}, int = 0); // #default_lambda2
void trigger() { foo(0, [](int){}); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#default_lambda1 {{candidate function}}
// expected-note@#default_lambda2 {{candidate function}}
}

namespace lambda_return_decltype {
template <class T> void foo(T a, auto f = []{ return decltype(a){}; }); // #lambda_return_decltype1
template <class T> void foo(T a, auto f = []{ return decltype(a){}; }, int = 0); // #lambda_return_decltype2
void trigger() { foo(0, []{ return 0; }); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#lambda_return_decltype1 {{candidate function}}
// expected-note@#lambda_return_decltype2 {{candidate function}}
}

namespace decltype_blocks {
template <class T> void foo(T a, void (^)(decltype(a))); // #decltype_blocks1
template <class T> void foo(T a, void (^)(decltype(a)), int = 0); // #decltype_blocks2
void trigger() { foo(0, (void (^)(int))0); }
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#decltype_blocks1 {{candidate function}}
// expected-note@#decltype_blocks2 {{candidate function}}
}
