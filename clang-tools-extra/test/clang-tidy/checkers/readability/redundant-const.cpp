// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-const %t

struct Foo {};

constexpr int n1 = 10;
const int n2 = 20;
constexpr Foo n3 = {};

constexpr const int p1 = 10;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p1 = 10;

const constexpr int p2 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p2 = 20;

static const constexpr int p3 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: static constexpr int p3 = 20;

constexpr const Foo p4 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr Foo p4 = {};

// Since constexpr makes only the pointer const, this usage is not redundant.
constexpr const char* n4 = "hello";

constexpr const auto n5 = "hello";

constexpr const auto const n6 = "hello";

constexpr const char* const p5 = "hello";
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const char* p5 = "hello";

template<typename T>
const constexpr T p6 = {};
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr T p6 = {};

constexpr const int* n7 = p6<int*>;

const constexpr double p7 = p6<double>;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr double p7 = p6<double>;

constexpr const int* const p8 = p6<int*>;
// CHECK-MESSAGES: [[@LINE-1]]:22: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int* p8 = p6<int*>;

void f() {
  constexpr Foo n1 = {};
  const Foo n2 = {};

  const constexpr Foo p1 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: redundant use of 'const'; 'constexpr' already implies 'const'
  // CHECK-FIXES: constexpr Foo p1 = {};

  static const constexpr Foo p2 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: redundant use of 'const'; 'constexpr' already implies 'const'
  // CHECK-FIXES: static constexpr Foo p2 = {};
}

struct Config {
    static const constexpr bool p = false;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant use of 'const'; 'constexpr' already implies 'const'
    // CHECK-FIXES: static constexpr bool p = false;
};

template <typename T>
class Templated {
    static const constexpr int size = 10;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant use of 'const'; 'constexpr' already implies 'const'
    // CHECK-FIXES: static constexpr int size = 10;
    int data[size];
};

constexpr Templated<int> n8{};

template <int N>
struct Templated2 {
    static const constexpr int size = N;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant use of 'const'; 'constexpr' already implies 'const'
    // CHECK-FIXES: static constexpr int size = N;
    int data[size];
};

static constexpr int n9[] = {0, 1, 4, 9, 16};

constexpr const int p9[] = {0, 1, 4, 9, 16};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p9[] = {0, 1, 4, 9, 16};

constexpr int square(int n) { return n * n; }

const constexpr int p10 = square(10);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p10 = square(10);

constexpr int n10 = square(5);

constexpr Foo** n11 = nullptr;

constexpr Foo* const* n12 = nullptr;

constexpr Foo* const* const p11 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr Foo* const* p11 = nullptr;

constexpr const Foo* const* const p12 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:29: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const Foo* const* p12 = nullptr;

constexpr const int (*n13)[10] = nullptr;

constexpr const int (*const p13)[10] = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int (*p13)[10] = nullptr;

constexpr int (*n14)(int) = nullptr;

constexpr int (*const p14)(int) = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:17: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int (*p14)(int) = nullptr;

struct Bar {
    int x, y;
    int sum() { return x + y; }
};

constexpr const int Bar::*n15 = &Bar::x;

constexpr const int Bar::* const p15 = &Bar::x;
// CHECK-MESSAGES: [[@LINE-1]]:28: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int Bar::* p15 = &Bar::x;

constexpr int (Bar::*n16)() = &Bar::sum;

constexpr int (Bar::* const p16)() = &Bar::sum;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int (Bar::* p16)() = &Bar::sum;

#define CONSTEXPR constexpr
#define CONST const

CONSTEXPR Foo n17 = {};

CONSTEXPR const Foo p17 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: CONSTEXPR Foo p17 = {};

const CONSTEXPR Foo p18 = {};
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: CONSTEXPR Foo p18 = {};

CONST constexpr Foo n18 = {};
constexpr CONST Foo n19 = {};

const Foo* n20 = nullptr;

// OK for references
constexpr const Foo& n21 = p18;
constexpr const Foo*& n22 = n20;

constexpr const decltype(nullptr) p19 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr decltype(nullptr) p19 = nullptr;

constexpr const decltype(0) p20 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr decltype(0) p20 = {};
