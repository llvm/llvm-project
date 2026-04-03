// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-const %t

struct Foo {};

// Simple allowed usages, nothing to warn
constexpr int n1 = 10;
const int n2 = 20;
constexpr Foo n3 = {};

constexpr const int p1 = 10;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int p1 = 10;

const constexpr int p2 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int p2 = 20;

static const constexpr int p3 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: static constexpr int p3 = 20;

constexpr const Foo p4 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr Foo p4 = {};

// Since constexpr makes only the pointer const, this usage is not redundant.
constexpr const char* n4 = "hello";

constexpr const char* const n5 = "hello";
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr const char* n5 = "hello";

// Since T might be a pointer type, we don't warn on this.
template<typename T>
const constexpr T n6 = {};

constexpr const int* n7 = n6<int*>;

const constexpr double p5 = n6<double>;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr double p5 = n6<double>;

constexpr const int* const p6 = n6<int*>;
// CHECK-MESSAGES: [[@LINE-1]]:22: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr const int* p6 = n6<int*>;

void f() {
  constexpr Foo n1 = {};
  const Foo n2 = {};

  const constexpr Foo p1 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: redundant 'const' in constexpr variable declaration
  // CHECK-FIXES: constexpr Foo p1 = {};

  static const constexpr Foo p2 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: redundant 'const' in constexpr variable declaration
  // CHECK-FIXES: static constexpr Foo p2 = {};
}

struct Config {
    static const constexpr bool p = false;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant 'const' in constexpr variable declaration
    // CHECK-FIXES: static constexpr bool p = false;
};

template <typename T>
class Templated {
    static const constexpr int size = 10;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant 'const' in constexpr variable declaration
    // CHECK-FIXES: static constexpr int size = 10;
    int data[size];
};

constexpr Templated<int> b{};

template <int N>
struct Templated2 {
    static const constexpr int size = N;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant 'const' in constexpr variable declaration
    // CHECK-FIXES: static constexpr int size = N;
    int data[size];
};

static constexpr int n8[] = {0, 1, 4, 9, 16};

constexpr const int p7[] = {0, 1, 4, 9, 16};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int p7[] = {0, 1, 4, 9, 16};

constexpr int square(int n) { return n * n; }

const constexpr int p8 = square(10);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int p8 = square(10);

constexpr int n9 = square(5);

constexpr Foo** n10 = nullptr;

constexpr Foo* const* n11 = nullptr;

constexpr Foo* const* const p9 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr Foo* const* p9 = nullptr;

constexpr const Foo* const* const p10 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:29: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr const Foo* const* p10 = nullptr;

constexpr const int (*n12)[10] = nullptr;

constexpr const int (*const p11)[10] = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr const int (*p11)[10] = nullptr;

constexpr int (*n13)(int) = nullptr;

constexpr int (*const p12)(int) = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:17: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int (*p12)(int) = nullptr;

struct Bar {
    int x, y;
    int sum() { return x + y; }
};

// Pointer to data member
constexpr const int Bar::*n14 = &Bar::x;

// Pointer to data member
constexpr const int Bar::* const p13 = &Bar::x;
// CHECK-MESSAGES: [[@LINE-1]]:28: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr const int Bar::* p13 = &Bar::x;

// Pointer to member function
constexpr int (Bar::*n15)() = &Bar::sum;

// Pointer to member function
constexpr int (Bar::* const p14)() = &Bar::sum;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: constexpr int (Bar::* p14)() = &Bar::sum;

#define CONSTEXPR constexpr

CONSTEXPR Foo n16 = {};

CONSTEXPR const Foo p15 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: CONSTEXPR Foo p15 = {};

const CONSTEXPR Foo p16 = {};
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant 'const' in constexpr variable declaration
// CHECK-FIXES: CONSTEXPR Foo p16 = {};

// OK for references
constexpr const Foo& n17 = p15;
