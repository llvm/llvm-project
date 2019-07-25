// RUN: %clang_cc1 -std=c++17 -O0 %s -emit-llvm -o /dev/null -verify -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++17 -O0 %s -emit-llvm -o /dev/null -verify -triple %ms_abi_triple

// Minimal reproducer for PR42665.
// expected-no-diagnostics

struct Foo {
  Foo() = default;
  virtual ~Foo() = default;
};

template <typename Deleter>
struct Pair {
  Foo first;
  Deleter second;
};

template <typename Deleter>
Pair(Foo, Deleter) -> Pair<Deleter>;

template <typename T>
void deleter(T& t) { t.~T(); }

auto make_pair() {
  return Pair{ Foo(), deleter<Foo> };
}

void foobar() {
  auto p = make_pair();
  auto& f = p.first;
  auto& d = p.second;
  d(f); // Invoke virtual destructor of Foo through d.
} // p's destructor is invoked.

