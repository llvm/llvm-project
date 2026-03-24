// RUN: %clangxx -std=c++20 -fsyntax-only %s

#include <initializer_list>

// Plain `auto` cases covered by the fast path.
int i = 0;
int &r = i;
auto a = r;
static_assert(__is_same(decltype(a), int));

const int ci = 0;
auto b = ci;
static_assert(__is_same(decltype(b), int));

const int &cr = ci;
auto c = cr;
static_assert(__is_same(decltype(c), int));

int arr[3];
auto arr_decay = arr;
static_assert(__is_same(decltype(arr_decay), int *));

void foo();
auto func_decay = foo;
static_assert(__is_same(decltype(func_decay), void (*)(void)));

int *const p = nullptr;
auto ptr_top_const_removed = p;
static_assert(__is_same(decltype(ptr_top_const_removed), int *));

const int *q = nullptr;
auto ptr_pointee_const_preserved = q;
static_assert(__is_same(decltype(ptr_pointee_const_preserved), const int *));

int arr2[3] = {0, 1, 2};
int (&rarr)[3] = arr2;
auto array_ref_decay = rarr;
static_assert(__is_same(decltype(array_ref_decay), int *));

auto str_decay = "abc";
static_assert(__is_same(decltype(str_decay), const char *));

int arr3[2][3] = {{1, 2, 3}, {4, 5, 6}};
auto multi_arr_decay = arr3;
static_assert(__is_same(decltype(multi_arr_decay), int (*)[3]));

volatile int vi = 0;
auto volatile_value = vi;
static_assert(__is_same(decltype(volatile_value), int));

volatile int *vp = nullptr;
auto volatile_ptr = vp;
static_assert(__is_same(decltype(volatile_ptr), volatile int *));

// Non-fast-path init-list case should remain unchanged.
auto ilist = {1, 2, 3};
static_assert(__is_same(decltype(ilist), std::initializer_list<int>));

// Reference-valued initializers.
int j = 1;
int &lr = j;
int &&rr = 2;

auto ref_lvalue = lr;
static_assert(__is_same(decltype(ref_lvalue), int));

// A named rvalue reference is still an lvalue expression.
auto ref_named_rvalue = rr;
static_assert(__is_same(decltype(ref_named_rvalue), int));

const int cj = 3;
const int &clr = cj;
auto ref_const_lvalue = clr;
static_assert(__is_same(decltype(ref_const_lvalue), int));

const int &&crr = 4;
auto ref_const_rvalue = crr;
static_assert(__is_same(decltype(ref_const_rvalue), int));

void (&func_ref)() = foo;
auto func_ref_decay = func_ref;
static_assert(__is_same(decltype(func_ref_decay), void (*)(void)));

// Adjacent `const auto` cases should remain unchanged.
const auto ca1 = 0;
static_assert(__is_same(decltype(ca1), const int));

int i2 = 1;
int &lr2 = i2;
const auto ca2 = lr2;
static_assert(__is_same(decltype(ca2), const int));

const int ci2 = 2;
const auto ca3 = ci2;
static_assert(__is_same(decltype(ca3), const int));

int arr4[3] = {1, 2, 3};
const auto ca4 = arr4;
static_assert(__is_same(decltype(ca4), int *const));

void qux();
const auto ca5 = qux;
static_assert(__is_same(decltype(ca5), void (*const)(void)));


