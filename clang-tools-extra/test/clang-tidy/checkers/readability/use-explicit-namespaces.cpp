// RUN: %check_clang_tidy %s readability-use-explicit-namespaces %t

namespace foo {
void doSomething() {}
} // namespace foo

void test1() { foo::doSomething(); }

using namespace foo;

void test2() {
  doSomething();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
}
