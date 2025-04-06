// RUN: %clang_cc1 -o - -emit-llvm -triple x86_64-linux-gnu %s

// Check there are no crash issue CodeGen action.
// https://github.com/llvm/llvm-project/pull/97308
struct a {
} constexpr b;
class c {
public:
  c(a);
};
class B {
public:
  using d = int;
  struct e {
    enum { f } g;
    int h;
    c i;
    d j{};
  };
};
B::e k{B::e::f, int(), b};
