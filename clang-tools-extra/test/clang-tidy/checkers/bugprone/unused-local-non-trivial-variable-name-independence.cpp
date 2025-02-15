// RUN: %check_clang_tidy -std=c++23 -check-suffixes=,CXX23 %s bugprone-unused-local-non-trivial-variable %t -- \
// RUN:       -config="{CheckOptions: {bugprone-unused-local-non-trivial-variable.IncludeTypes: '::async::Foo'}}" \
// RUN:       --
// RUN: %check_clang_tidy -std=c++26 %s bugprone-unused-local-non-trivial-variable %t -- \
// RUN:       -config="{CheckOptions: {bugprone-unused-local-non-trivial-variable.IncludeTypes: '::async::Foo'}}" \
// RUN:       --

namespace async {
class Foo {
  public:
    ~Foo();
  private:
};
} // namespace async

void check() {
  async::Foo C;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: unused local variable 'C' of type 'async::Foo' [bugprone-unused-local-non-trivial-variable]
  async::Foo _;
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: unused local variable '_' of type 'async::Foo' [bugprone-unused-local-non-trivial-variable]
}
