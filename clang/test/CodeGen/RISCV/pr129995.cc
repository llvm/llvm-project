// RUN: %clang_cc1 triple riscv64 -emit-llvm -target-feature +m -target-feature +v -target-abi lp64d -o /dev/null %s

struct a {
  using b = char __attribute__((vector_size(sizeof(char))));
};
class c {
  using d = a::b;
  d e;

public:
  static c f();
};
class g {
public:
  template <class h> g(h);
  friend g operator^(g, g) { c::f; }
  friend g operator^=(g i, g j) { i ^ j; }
};
template <typename, int> using k = g;
template <typename l> using m = k<l, sizeof(l)>;
void n() {
  void o();
  m<char> p = o ^= p;
}
