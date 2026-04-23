// RUN: %clang_cc1 -std=c++20 -fblocks -triple x86_64-apple-darwin %s -verify

template<void (^B)()>
struct A {
  void call() { B(); }
};

constexpr void (^global_block)() = ^{};
void test_global() {
  A<global_block> a;
  a.call();
}

void test_literal() {
  A<^{}> a;
  a.call();
}

void test_null() {
  A<nullptr> a;
}

void test_capturing(int x) {
  A<^{ (void)x; }> a; // expected-error {{non-type template argument is not a constant expression}}
}

template<void (^B)()>
void deduce(A<B> a) {}

void test_deduce() {
  A<global_block> a;
  deduce(a);
}

constexpr void (^another_block)() = ^{};
static_assert(!__is_same(A<global_block>, A<another_block>));

template<int (^B)(int)>
struct BFunc {
  int call(int x) { return B(x); }
};

void test_params() {
  BFunc<^(int x) { return x + 1; }> b;
  (void)b.call(1);
}


template<auto B>
struct AutoBlock {};

template<auto B>
void deduce_auto(AutoBlock<B> a) {}

void test_auto() {
  AutoBlock<global_block> a;
  deduce_auto(a);
}

template<typename T, T B>
struct TypedBlock {};

template<typename T, T B>
void deduce_type(TypedBlock<T, B> a) {}

void test_typed() {
  TypedBlock<void(^)(), global_block> a;
  deduce_type(a);
}
