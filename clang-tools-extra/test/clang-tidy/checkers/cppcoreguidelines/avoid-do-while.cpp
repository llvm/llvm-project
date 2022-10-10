// RUN: %check_clang_tidy -check-suffixes=DEFAULT       %s cppcoreguidelines-avoid-do-while %t
// RUN: %check_clang_tidy -check-suffixes=IGNORE-MACROS %s cppcoreguidelines-avoid-do-while %t -- -config='{CheckOptions: [{key: cppcoreguidelines-avoid-do-while.IgnoreMacros, value: true}]}'

#define FOO(x) \
  do { \
  } while (x != 0)

#define BAR_0(x) \
  do { \
    bar(x); \
  } while (0)

#define BAR_FALSE(x) \
  do { \
    bar(x); \
  } while (false)

void bar(int);
int baz();

void foo()
{
    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops [cppcoreguidelines-avoid-do-while]
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops [cppcoreguidelines-avoid-do-while]
    do {

    } while(0);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    do {

    } while(1);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    do {

    } while(-1);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    do {

    } while(false);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    do {

    } while(true);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+3]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+2]]:5: warning: avoid do-while loops
    int x1{0};
    do {
      x1 = baz();
    } while (x1 > 0);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+2]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    do {

    } while (x1 != 0);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+3]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+2]]:5: warning: avoid do-while loops
    constexpr int x2{0};
    do {

    } while (x2);

    // CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE+3]]:5: warning: avoid do-while loops
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+2]]:5: warning: avoid do-while loops
    constexpr bool x3{false};
    do {

    } while (x3);

    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    FOO(x1);

    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    BAR_0(x1);

    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: warning: avoid do-while loops
    BAR_FALSE(x1);
}
