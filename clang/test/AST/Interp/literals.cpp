// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify=ref %s

static_assert(true, "");
static_assert(false, ""); // expected-error{{failed}} ref-error{{failed}}
static_assert(nullptr == nullptr, "");
static_assert(1 == 1, "");
static_assert(1 == 3, ""); // expected-error{{failed}} ref-error{{failed}}

constexpr int number = 10;
static_assert(number == 10, "");
static_assert(number != 10, ""); // expected-error{{failed}} \
                                 // ref-error{{failed}} \
                                 // ref-note{{evaluates to}}

constexpr bool getTrue() { return true; }
constexpr bool getFalse() { return false; }
constexpr void* getNull() { return nullptr; }

constexpr int neg(int m) { return -m; }
constexpr bool inv(bool b) { return !b; }

static_assert(12, "");
static_assert(12 == -(-(12)), "");
static_assert(!false, "");
static_assert(!!true, "");
static_assert(!!true == !false, "");
static_assert(true == 1, "");
static_assert(false == 0, "");
static_assert(!5 == false, "");
static_assert(!0, "");
static_assert(-true, "");
static_assert(-false, ""); //expected-error{{failed}} ref-error{{failed}}

constexpr int m = 10;
constexpr const int *p = &m;
static_assert(p != nullptr, "");
static_assert(*p == 10, "");

constexpr const int* getIntPointer() {
  return &m;
}
//static_assert(getIntPointer() == &m, ""); TODO
//static_assert(*getIntPointer() == 10, ""); TODO

constexpr int gimme(int k) {
  return k;
}
// static_assert(gimme(5) == 5, ""); TODO
