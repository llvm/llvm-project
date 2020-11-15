// RUN:  %clang_cc1 -std=c++2a -verify %s
template<typename T, typename U> constexpr bool is_same_v = false;
template<typename T> constexpr bool is_same_v<T, T> = true;

template<typename... T>
struct type_list;

namespace unconstrained {
  decltype(auto) f1(auto x) { return x; }
  static_assert(is_same_v<decltype(f1(1)), int>);
  static_assert(is_same_v<decltype(f1('c')), char>);

  decltype(auto) f2(auto &x) { return x; }
  // expected-note@-1{{candidate function [with x:auto = int] not viable: expects an lvalue for 1st argument}}
  // expected-note@-2{{candidate function [with x:auto = char] not viable: expects an lvalue for 1st argument}}
  static_assert(is_same_v<decltype(f2(1)), int &>); // expected-error{{no matching}}
  static_assert(is_same_v<decltype(f2('c')), char &>); // expected-error{{no matching}}

  decltype(auto) f3(const auto &x) { return x; }
  static_assert(is_same_v<decltype(f3(1)), const int &>);
  static_assert(is_same_v<decltype(f3('c')), const char &>);

  decltype(auto) f4(auto (*x)(auto y)) { return x; } // expected-error{{'auto' not allowed in function prototype}}

  decltype(auto) f5(void (*x)(decltype(auto) y)) { return x; } // expected-error{{'decltype(auto)' not allowed in function prototype}}

  int return_int(); void return_void(); int foo(int);

  decltype(auto) f6(auto (*x)()) { return x; }
  // expected-note@-1{{candidate template ignored: failed template argument deduction}}
  static_assert(is_same_v<decltype(f6(return_int)), int (*)()>);
  static_assert(is_same_v<decltype(f6(return_void)), void (*)()>);
  using f6c1 = decltype(f6(foo)); // expected-error{{no matching}}

  decltype(auto) f7(auto (*x)() -> int) { return x; }
  // expected-note@-1{{candidate function not viable: no known conversion from 'void ()' to 'auto (*)() -> int' for 1st argument}}
  // expected-note@-2{{candidate function not viable: no known conversion from 'int (int)' to 'auto (*)() -> int' for 1st argument}}
  static_assert(is_same_v<decltype(f7(return_int)), int (*)()>);
  using f7c1 = decltype(f7(return_void)); // expected-error{{no matching}}
  using f7c2 = decltype(f7(foo)); // expected-error{{no matching}}
  static_assert(is_same_v<decltype(&f7), int (*(*)(int (*x)()))()>);

  decltype(auto) f8(auto... x) { return (x + ...); }
  static_assert(is_same_v<decltype(f8(1, 2, 3)), int>);
  static_assert(is_same_v<decltype(f8('c', 'd')), int>);
  static_assert(is_same_v<decltype(f8('c', 1)), int>);

  decltype(auto) f9(auto &... x) { return (x, ...); }
  // expected-note@-1{{candidate function [with x:auto = <int (), int>] not viable: expects an lvalue for 2nd argument}}
  using f9c1 = decltype(f9(return_int, 1)); // expected-error{{no matching}}

  decltype(auto) f11(decltype(auto) x) { return x; } // expected-error{{'decltype(auto)' not allowed in function prototype}}

  template<typename T>
  auto f12(auto x, T y) -> type_list<T, decltype(x)>;
  static_assert(is_same_v<decltype(f12(1, 'c')), type_list<char, int>>);
  static_assert(is_same_v<decltype(f12<char>(1, 'c')), type_list<char, int>>);

  template<typename T>
  auto f13(T x, auto y) -> type_list<T, decltype(y)>;
  static_assert(is_same_v<decltype(f13(1, 'c')), type_list<int, char>>);
  static_assert(is_same_v<decltype(f13<char>(1, 'c')), type_list<char, char>>);

  template<typename T>
  auto f14(auto y) -> type_list<T, decltype(y)>;
  static_assert(is_same_v<decltype(f14<int>('c')), type_list<int, char>>);
  static_assert(is_same_v<decltype(f14<int, char>('c')), type_list<int, char>>);

  template<typename T, typename U>
  auto f15(auto y, U u) -> type_list<T, U, decltype(y)>;
  static_assert(is_same_v<decltype(f15<int>('c', nullptr)), type_list<int, decltype(nullptr), char>>);
  static_assert(is_same_v<decltype(f15<int, decltype(nullptr)>('c', nullptr)), type_list<int, decltype(nullptr), char>>);

  auto f16(auto x, auto y) -> type_list<decltype(x), decltype(y)>;
  static_assert(is_same_v<decltype(f16('c', 1)), type_list<char, int>>);
  static_assert(is_same_v<decltype(f16<int>('c', 1)), type_list<int, int>>);
  static_assert(is_same_v<decltype(f16<int, char>('c', 1)), type_list<int, char>>);

  void f17(auto x, auto y) requires (sizeof(x) > 1);
  // expected-note@-1{{candidate template ignored: constraints not satisfied [with x:auto = char, y:auto = int]}}
  // expected-note@-2{{because 'sizeof (x) > 1' (1 > 1) evaluated to false}}
  static_assert(is_same_v<decltype(f17('c', 1)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f17<int>('c', 1)), void>);
  static_assert(is_same_v<decltype(f17<int, char>('c', 1)), void>);

  void f18(auto... x) requires (sizeof...(x) == 2);
  // expected-note@-1{{candidate template ignored: constraints not satisfied [with x:auto = <char, int, int>]}}
  // expected-note@-2{{candidate template ignored: constraints not satisfied [with x:auto = <char>]}}
  // expected-note@-3{{because 'sizeof...(x) == 2' (1 == 2) evaluated to false}}
  // expected-note@-4{{because 'sizeof...(x) == 2' (3 == 2) evaluated to false}}
  static_assert(is_same_v<decltype(f18('c')), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f18('c', 1)), void>);
  static_assert(is_same_v<decltype(f18('c', 1, 2)), void>);
  // expected-error@-1{{no matching}}

  template<typename T>
  struct S {
    constexpr auto f1(auto x, T t) -> decltype(x + t);

    template<typename U>
    constexpr auto f2(U u, auto x, T t) -> decltype(x + u + t);
  };

  template<typename T>
  constexpr auto S<T>::f1(auto x, T t) -> decltype(x + t) { return x + t; }

  template<typename T>
  template<typename U>
  constexpr auto S<T>::f2(auto x, U u, T t) -> decltype(x + u + t) { return x + u + t; }
  // expected-error@-1 {{out-of-line definition of 'f2' does not match any declaration in 'S<T>'}}

  template<typename T>
  template<typename U>
  constexpr auto S<T>::f2(U u, auto x, T t) -> decltype(x + u + t) { return x + u + t; }

  template<>
  template<>
  constexpr auto S<int>::f2<double>(double u, char x, int t) -> double { return 42; }

  static_assert(S<char>{}.f1(1, 2) == 3);
  static_assert(S<char>{}.f2(1, 2, '\x00') == 3);
  static_assert(S<char>{}.f2<double>(1, 2, '\x00') == 3.);
  static_assert(S<int>{}.f2<double>(1, '2', '\x00') == 42);
}

namespace constrained {
  template<typename T>
  concept C = is_same_v<T, int>;
  // expected-note@-1 12{{because}}
  template<typename T, typename U>
  concept C2 = is_same_v<T, U>;
  // expected-note@-1 12{{because}}

  int i;
  const int ci = 1;
  char c;
  const char cc = 'a';
  int g(int);
  char h(int);

  void f1(C auto x);
  // expected-note@-1 {{candidate template ignored: constraints not satisfied [with x:auto = }}
  // expected-note@-2{{because}}
  static_assert(is_same_v<decltype(f1(1)), void>);
  static_assert(is_same_v<decltype(f1('a')), void>);
  // expected-error@-1{{no matching}}
  void f2(C auto &x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f2(i)), void>);
  static_assert(is_same_v<decltype(f2(ci)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f2(c)), void>);
  // expected-error@-1{{no matching}}
  void f3(const C auto &x);
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f3(i)), void>);
  static_assert(is_same_v<decltype(f3(ci)), void>);
  static_assert(is_same_v<decltype(f3(c)), void>);
  // expected-error@-1{{no matching}}
  void f4(C auto (*x)(C auto y)); // expected-error{{'auto' not allowed}}
  void f5(C auto (*x)(int y));
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f5(g)), void>);
  static_assert(is_same_v<decltype(f5(h)), void>);
  // expected-error@-1{{no matching}}
  void f6(C auto (*x)() -> int); // expected-error{{function with trailing return type must specify return type 'auto', not 'C auto'}}
  void f7(C auto... x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f7(1, 2)), void>);
  static_assert(is_same_v<decltype(f7(1, 'a')), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f7('a', 2)), void>);
  // expected-error@-1{{no matching}}
  void f8(C auto &... x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f8(i, i)), void>);
  static_assert(is_same_v<decltype(f8(i, c)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f8(i, i, ci)), void>);
  // expected-error@-1{{no matching}}
  void f9(const C auto &... x);
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f9(i, i)), void>);
  static_assert(is_same_v<decltype(f9(i, c)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f9(i, i, ci)), void>);
  void f10(C decltype(auto) x);
  auto f11 = [] (C auto x) { };
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f11(1)), void>);
  static_assert(is_same_v<decltype(f11('a')), void>);
  // expected-error@-1{{no matching}}

  void f12(C2<char> auto x);
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f12(1)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f12('a')), void>);
  void f13(C2<char> auto &x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f13(i)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f13(cc)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f13(c)), void>);
  void f14(const C2<char> auto &x);
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f14(i)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f14(cc)), void>);
  static_assert(is_same_v<decltype(f14(c)), void>);
  void f15(C2<char> auto (*x)(C2<int> auto y)); // expected-error{{'auto' not allowed}}
  void f16(C2<char> auto (*x)(int y));
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f16(g)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f16(h)), void>);
  void f17(C2<char> auto (*x)() -> int); // expected-error{{function with trailing return type must specify return type 'auto', not 'C2<char> auto'}}
  void f18(C2<char> auto... x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f18('a', 'b')), void>);
  static_assert(is_same_v<decltype(f18('a', 1)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f18(2, 'a')), void>);
  // expected-error@-1{{no matching}}
  void f19(C2<char> auto &... x);
  // expected-note@-1 2{{candidate template ignored}} expected-note@-1 2{{because}}
  static_assert(is_same_v<decltype(f19(c, c)), void>);
  static_assert(is_same_v<decltype(f19(i, c)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f19(c, c, cc)), void>);
  // expected-error@-1{{no matching}}
  void f20(const C2<char> auto &... x);
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f20(c, c)), void>);
  static_assert(is_same_v<decltype(f20(i, c)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f20(c, c, cc)), void>);
  void f21(C2<char> decltype(auto) x);
  auto f22 = [] (C2<char> auto x) { };
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  static_assert(is_same_v<decltype(f22(1)), void>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(f22('a')), void>);

  struct S1 { S1(C auto); };
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  // expected-note@-2 2{{candidate constructor}}
  static_assert(is_same_v<decltype(S1(1)), S1>);
  static_assert(is_same_v<decltype(S1('a')), S1>);
  // expected-error@-1{{no matching}}
  struct S2 { S2(C2<char> auto); };
  // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
  // expected-note@-2 2{{candidate constructor}}
  static_assert(is_same_v<decltype(S2(1)), S2>);
  // expected-error@-1{{no matching}}
  static_assert(is_same_v<decltype(S2('a')), S2>);
}
