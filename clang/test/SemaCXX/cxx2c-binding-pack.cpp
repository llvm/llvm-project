// RUN: %clang_cc1 -fsyntax-only -std=c++26 %s -verify

template <typename T>
struct type_ { };

template <typename ...T>
auto sum(T... t) { return (t + ...); }

struct my_struct {
  int a;
  int b;
  int c;
  int d;
};

struct fake_tuple {
  int arr[4] = {1, 2, 3, 6};

  template <unsigned i>
  int get() {
    return arr[i];
  }
};

namespace std {
  template <typename T>
  struct tuple_size;
  template <unsigned i, typename T>
  struct tuple_element;

  template <>
  struct tuple_size<fake_tuple> {
    static constexpr unsigned value = 4;
  };

  template <unsigned i>
  struct tuple_element<i, fake_tuple> {
    using type = int;
  };
}


template <typename T>
void decompose_tuple() {
  auto tup = T{{1, 2, 3, 6}};
  auto&& [x, ...rest, y] = tup;

  ((void)type_<int>(type_<decltype(rest)>{}), ...);

  T arrtup[2] = {T{{1, 2, 3, 6}},
                 T{{7, 9, 10, 11}}};
  int sum = 0;
  for (auto [...xs] : arrtup) {
    sum += (xs + ...);
  }
}

template <typename T>
void decompose_struct() {
  T obj{1, 2, 3, 6};
  auto [x, ...rest, y] = obj;

  auto [...empty] = type_<int>{};
  static_assert(sizeof...(empty) == 0);
}

template <typename T>
void decompose_array() {
  int arr[4] = {1, 2, 3, 6};
  auto [x, ...rest, y] = arr;

  static_assert(sizeof...(rest) == 2);
  int size = sizeof...(rest);
  T arr2[sizeof...(rest)] = {rest...};
  auto [...pack] = arr2;

  // Array of size 1.
  int arr1[1] = {1};
  auto [a, ...b] = arr1;
  static_assert(sizeof...(b) == 0);
  auto [...c] = arr1;
  static_assert(sizeof...(c) == 1);
  auto [a1, ...b1, c1] = arr1; // expected-error{{decomposes into 1 element, but 3 names were provided}}
}

// Test case by Younan Zhang.
template <unsigned... P>
struct S {
  template <unsigned... Q>
  struct N {
    void foo() {
      int arr[] = {P..., Q...};
      auto [x, y, ...rest] = arr;
      [&]() {
        static_assert(sizeof...(rest) + 2 == sizeof...(P) + sizeof...(Q));
      }();
    }
  };
};

struct bit_fields {
  int a : 4 {1};
  int b : 4 {2};
  int c : 4 {3};
  int d : 4 {4};
};

template <typename T>
void decompose_bit_field() {
  auto [...x] = T{};
  static_assert(sizeof...(x) == 4);
  int a = x...[0];
  int b = x...[1];
  int c = x...[2];
  int d = x...[3];
}

template <typename T>
void lambda_capture() {
  auto [...x] = T{};
  [=] { (void)sum(x...); }();
  [&] { (void)sum(x...); }();
  [x...] { (void)sum(x...); }();
  [&x...] { (void)sum(x...); }();
}

int main() {
  decompose_array<int>();
  decompose_tuple<fake_tuple>();
  decompose_struct<my_struct>();
  S<1, 2, 3, 4>::N<5, 6>().foo();
  decompose_bit_field<bit_fields>();
  lambda_capture<int[5]>();
  lambda_capture<fake_tuple>();
  lambda_capture<my_struct>();
}

// P1061R10 Stuff
namespace {
struct C { int x, y, z; };

template <class T>
void now_i_know_my() {
  auto [a, b, c] = C(); // OK, SB0 is a, SB1 is b, and SB2 is c
  auto [d, ...e] = C(); // OK, SB0 is d, the pack e (v1) contains two structured bindings: SB1 and SB2
  static_assert(sizeof...(e) == 2);
  auto [...f, g] = C(); // OK, the pack f (v0) contains two structured bindings: SB0 and SB1, and SB2 is g
  static_assert(sizeof...(e) == 2);
  auto [h, i, j, ...k] = C(); // OK, the pack k is empty
  static_assert(sizeof...(e) == 0);
  auto [l, m, n, o, ...p] = C(); // expected-error{{{decomposes into 3 elements, but 5 names were provided}}}
}
}  // namespace

namespace {
auto g() -> int(&)[4];

template <unsigned long N>
void h(int (&arr)[N]) {
  auto [a, ...b, c] = arr;  // a names the first element of the array,
                            // b is a pack referring to the second and
                            // third elements, and c names the fourth element
  static_assert(sizeof...(b) == 2);
  auto& [...e] = arr;        // e is a pack referring to the four elements of the array
  static_assert(sizeof...(e) == 4);
}

void call_h() {
 h(g());
}
}  // namespace

namespace {
struct D { };

int g(...) { return 1; }

template <typename T>
constexpr int f() {
  D arr[1];
  auto [...e] = arr;
  return g(e...);
}

constexpr int g(D) { return 2; }

void other_main() {
  static_assert(f<int>() == 2);
}
}  // namespace
