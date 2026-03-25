// RUN: %clangxx -std=c++20 -fsyntax-only -Xclang -verify %s

#include <initializer_list>


// Plain auto tests
int i = 0;
auto a = i;
static_assert(__is_same(decltype(a), int));

const int ci = 0;
auto b = ci;
static_assert(__is_same(decltype(b), int));

volatile int vi = 0;
auto c = vi;
static_assert(__is_same(decltype(c), int));

int &r = i;
auto d = r;
static_assert(__is_same(decltype(d), int));

const int &cr = ci;
auto e = cr;
static_assert(__is_same(decltype(e), int));

int &&rr = 1;
auto f = rr;
static_assert(__is_same(decltype(f), int));


// Plain auto with array / function

int arr[3];
auto arr_decay = arr;
static_assert(__is_same(decltype(arr_decay), int *));

int arr2[2][3];
auto arr2_decay = arr2;
static_assert(__is_same(decltype(arr2_decay), int (*)[3]));

int (&rarr)[3] = arr;
auto rarr_decay = rarr;
static_assert(__is_same(decltype(rarr_decay), int *));

void foo();
auto func_decay = foo;
static_assert(__is_same(decltype(func_decay), void (*)(void)));

void (&func_ref)() = foo;
auto func_ref_decay = func_ref;
static_assert(__is_same(decltype(func_ref_decay), void (*)(void)));

auto str_decay = "abc";
static_assert(__is_same(decltype(str_decay), const char *));


// pointer qualifier

int *ip = nullptr;
auto p1 = ip;
static_assert(__is_same(decltype(p1), int *));

int *const ipc = nullptr;
auto p2 = ipc;
static_assert(__is_same(decltype(p2), int *));

const int *cip = nullptr;
auto p3 = cip;
static_assert(__is_same(decltype(p3), const int *));

volatile int *vip = nullptr;
auto p4 = vip;
static_assert(__is_same(decltype(p4), volatile int *));

int * __restrict__ rp = nullptr;
auto p5 = rp;
static_assert(__is_same(decltype(p5), int *));

const int * __restrict__ crp = nullptr;
auto p6 = crp;
static_assert(__is_same(decltype(p6), const int *));

// non-canonical type
using Animal = int;
Animal animal = 0;
auto t1 = animal;
static_assert(__is_same(decltype(t1), Animal));

using AnimalPtr = int *;
AnimalPtr ap = nullptr;
auto t2 = ap;
static_assert(__is_same(decltype(t2), AnimalPtr));

using ConstInt = const int;
ConstInt cx = 0;
auto t3 = cx;
static_assert(__is_same(decltype(t3), int));

using IntArray3 = int[3];
IntArray3 ta = {1, 2, 3};
auto t4 = ta;
static_assert(__is_same(decltype(t4), int *));

using FuncTy = void();
FuncTy &fr = foo;
auto t5 = fr;
static_assert(__is_same(decltype(t5), void (*)(void)));

// class / union / enum

struct MyStruct {
  int x;
};
MyStruct sv{1};
auto class_value = sv;
static_assert(__is_same(decltype(class_value), MyStruct));

union MyUnion {
  int i;
  float f;
};
MyUnion uv{};
auto union_value = uv;
static_assert(__is_same(decltype(union_value), MyUnion));

enum MyEnum { EnumA, EnumB };
MyEnum ev = EnumA;
auto enum_value = ev;
static_assert(__is_same(decltype(enum_value), MyEnum));

enum class MyScopedEnum : unsigned long { X, Y };
MyScopedEnum sev = MyScopedEnum::X;
auto scoped_enum_value = sev;
static_assert(__is_same(decltype(scoped_enum_value), MyScopedEnum));


// const auto

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

using AliasConstInt = const int;
AliasConstInt aci = 3;
const auto ca6 = aci;
static_assert(__is_same(decltype(ca6), const int));

// auto *

int *ip1 = nullptr;
auto *ap1 = ip1;
static_assert(__is_same(decltype(ap1), int *));

const int *cip1 = nullptr;
auto *ap2 = cip1;
static_assert(__is_same(decltype(ap2), const int *));

int arr5[3];
auto *ap3 = arr5;
static_assert(__is_same(decltype(ap3), int *));

void f1();
auto *ap4 = f1;
static_assert(__is_same(decltype(ap4), void (*)(void)));

using Animal2 = int;
Animal2 *ap5 = nullptr;
auto *ap6 = ap5;
static_assert(__is_same(decltype(ap6), Animal2 *));


// const auto *

int *ip2 = nullptr;
const auto *cp1 = ip2;
static_assert(__is_same(decltype(cp1), const int *));

const int *cip2 = nullptr;
const auto *cp2 = cip2;
static_assert(__is_same(decltype(cp2), const int *));

int arr6[3];
const auto *cp3 = arr6;
static_assert(__is_same(decltype(cp3), const int *));

using Animal3 = int;
Animal3 *ap7 = nullptr;
const auto *cp4 = ap7;
static_assert(__is_same(decltype(cp4), const Animal3 *));


// auto ** / const auto **

int **pp1 = nullptr;
auto **dp1 = pp1;
static_assert(__is_same(decltype(dp1), int **));

const int **pp2 = nullptr;
auto **dp2 = pp2;
static_assert(__is_same(decltype(dp2), const int **));

const int *pbase = nullptr;
const int **pp3 = &pbase;
const auto **dp3 = pp3;
static_assert(__is_same(decltype(dp3), const int **));

using Animal4 = int;
Animal4 **pp4 = nullptr;
auto **dp4 = pp4;
static_assert(__is_same(decltype(dp4), Animal4 **));


// init-list

auto ilist = {1, 2, 3};
static_assert(__is_same(decltype(ilist), std::initializer_list<int>));

// untouched case


int x = 0;
auto *bad = x; // expected-error {{variable 'bad' with type 'auto *' has incompatible initializer of type 'int'}}

int y = 0;
auto &ref = y;
static_assert(__is_same(decltype(ref), int &));


int *p = nullptr;
auto *const pc = p;
static_assert(__is_same(decltype(pc), int * const));

const auto *const cpc = p;
static_assert(__is_same(decltype(cpc), const int * const));

