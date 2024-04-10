// RUN: %check_clang_tidy %s readability-static-definition-in-anonymous-namespace %t

namespace {

int a = 1;
const int b = 1;
static int c = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'c' is a static definition in anonymous namespace; static is redundant here [readability-static-definition-in-anonymous-namespace]
// CHECK-FIXES: {{^}}int c = 1;
static const int d = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'd' is a static definition in anonymous namespace
// CHECK-FIXES: {{^}}const int d = 1;
const static int e = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'e' is a static definition in anonymous namespace
// CHECK-FIXES: {{^}}const int e = 1;

void f() {
  int a = 1;
  static int b = 1;
}

static int g() {
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'g' is a static definition in anonymous namespace
// CHECK-FIXES: {{^}}int g() {
  return 1;
}

#define DEFINE_STATIC static
// CHECK-FIXES: {{^}}#define DEFINE_STATIC static
DEFINE_STATIC int h = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'h' is a static definition in anonymous namespace
// CHECK-FIXES: {{^}}DEFINE_STATIC int h = 1;

#define DEFINE_STATIC_VAR(x) static int x = 2
// CHECK-FIXES: {{^}}#define DEFINE_STATIC_VAR(x) static int x = 2
DEFINE_STATIC_VAR(i);
// CHECK-FIXES: {{^}}DEFINE_STATIC_VAR(i);

namespace inner {
int a = 1;
const int b = 1;
static int c = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'c' is a static definition in anonymous namespace; static is redundant here [readability-static-definition-in-anonymous-namespace]
// CHECK-FIXES: {{^}}int c = 1;
namespace deep_inner {
int a = 1;
const int b = 1;
static int c = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'c' is a static definition in anonymous namespace; static is redundant here [readability-static-definition-in-anonymous-namespace]
// CHECK-FIXES: {{^}}int c = 1;
} // namespace deep_inner
} // namespace inner

template<typename T>
static void printTemplate(T&&) {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'printTemplate' is a static definition in anonymous namespace; static is redundant here [readability-static-definition-in-anonymous-namespace]
// CHECK-FIXES: {{^}}void printTemplate(T&&) {}

void testTemplate() {
  printTemplate(5);
  printTemplate(5U);
  printTemplate("some string");
}

} // namespace

namespace N {

int a = 1;
const int b = 1;
static int c = 1;
static const int d = 1;
const static int e = 1;

} // namespace N
