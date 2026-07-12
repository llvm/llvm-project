// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-const %t

struct Foo {};

constexpr int n1 = 10;
const int n2 = 20;
constexpr Foo n3 = {};

constexpr const int p1 = 10;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p1 = 10;

constexpr int const p2 = 0;
// CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p2 = 0;

constexpr const int const p3 = 0;
// CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int p3 = 0;

const constexpr int p4 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p4 = 20;

static const constexpr int p5 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: static constexpr int p5 = 20;

constexpr const Foo p6 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr Foo p6 = {};

// Since constexpr makes only the pointer const, this usage is not redundant.
constexpr const char* n4 = "hello";

constexpr const auto n5 = "hello";

constexpr const auto const n6 = "hello";

constexpr const char* const p7 = "hello";
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const char* p7 = "hello";

template<typename T>
const constexpr T p8 = {};
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr T p8 = {};

constexpr const int* n7 = p8<int*>;

const constexpr double p9 = p8<double>;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr double p9 = p8<double>;

constexpr const int* const p10 = p8<int*>;
// CHECK-MESSAGES: [[@LINE-1]]:22: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int* p10 = p8<int*>;

void f() {
  constexpr Foo n1 = {};
  const Foo n2 = {};

  const constexpr Foo p1 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: redundant use of 'const'; 'constexpr' already implies 'const'
  // CHECK-FIXES: constexpr Foo p1 = {};

  static const constexpr Foo p4 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: redundant use of 'const'; 'constexpr' already implies 'const'
  // CHECK-FIXES: static constexpr Foo p4 = {};
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

constexpr const int p11[] = {0, 1, 4, 9, 16};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p11[] = {0, 1, 4, 9, 16};

constexpr int square(int n) { return n * n; }

const constexpr int p12 = square(10);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p12 = square(10);

constexpr int n10 = square(5);

constexpr Foo** n11 = nullptr;

constexpr Foo* const* n12 = nullptr;

constexpr Foo* const* const p13 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr Foo* const* p13 = nullptr;

constexpr const Foo* const* const p14 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:29: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const Foo* const* p14 = nullptr;

constexpr const int (*n13)[10] = nullptr;

constexpr const int (*const p15)[10] = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int (*p15)[10] = nullptr;

constexpr int (*n14)(int) = nullptr;

constexpr int (*const p16)(int) = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:17: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int (*p16)(int) = nullptr;

struct Bar {
    int x, y;
    int sum() { return x + y; }
};

constexpr const int Bar::*n15 = &Bar::x;

constexpr const int Bar::* const p17 = &Bar::x;
// CHECK-MESSAGES: [[@LINE-1]]:28: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const int Bar::* p17 = &Bar::x;

constexpr int (Bar::*n16)() = &Bar::sum;

constexpr int (Bar::* const p18)() = &Bar::sum;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int (Bar::* p18)() = &Bar::sum;

#define CONSTEXPR constexpr
#define CONST const

CONSTEXPR Foo n17 = {};

CONSTEXPR const Foo p19 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: CONSTEXPR Foo p19 = {};

const CONSTEXPR Foo p20 = {};
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: CONSTEXPR Foo p20 = {};

CONST constexpr Foo n18 = {};
constexpr CONST Foo n19 = {};

const Foo* n20 = nullptr;

// OK for references
constexpr const Foo& n21 = p20;
constexpr const Foo*& n22 = n20;

constexpr const decltype(nullptr) p21 = nullptr;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr decltype(nullptr) p21 = nullptr;

constexpr const decltype(0) p22 = {};
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr decltype(0) p22 = {};

constexpr const int p23 = 1, p24 = 2;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p23 = 1, p24 = 2;

// Reference sibling, warn without fixit.
constexpr const int p25 = 0, &p26 = p25;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

// Same for pointer siblings.
constexpr const int p27 = 0, *p28 = &p27;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

constexpr const int &p29 = p25, p30 = 0;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

constexpr const int p31 = 0,
                     &p32 = p31;
// CHECK-MESSAGES: [[@LINE-2]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

constexpr /* a */ const int p33 = 0, &p34 = p33;
// CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant use of 'const'; 'constexpr' already implies 'const'

constexpr const int p35 = 0, /* b */ &p36 = p35;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

constexpr /* c */ const int p37 = 1, p38 = 2;
// CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr /* c */ int p37 = 1, p38 = 2;

// Brace-initialized struct with reference sibling.
struct S { int x, y; };

constexpr const S p39 = {0, 1}, &p40 = p39;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'

// Rvalue reference sibling.
constexpr const int p41 = 0, &&p42 = 1;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
