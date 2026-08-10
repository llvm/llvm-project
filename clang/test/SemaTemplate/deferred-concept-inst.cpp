// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify -fsyntax-only -Wno-unused-value
// expected-no-diagnostics

namespace GithubBug44178 {
template <typename D>
struct CRTP {
  void call_foo()
    requires requires(D &v) { v.foo(); }
  {
    static_cast<D *>(this)->foo();
  }
};

struct Test : public CRTP<Test> {
  void foo() {}
};

int main() {
  Test t;
  t.call_foo();
  return 0;
}
} // namespace GithubBug44178
