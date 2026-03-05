// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Complex template metaprogramming with __builtin_is_constant_evaluated
//
// When using advanced template metaprogramming involving:
// - Template aliases with variadic templates
// - decltype expressions in template parameters
// - __builtin_is_constant_evaluated() in constexpr context
// - Complex member function template instantiation chains
// CIR fails during constant expression evaluation or template instantiation.

template <typename a> struct b {
  typedef a c;
};
template <typename a> class d {
public:
  typedef a c;
};
template <typename a> using ad = d<a>::c;
template <typename...> struct g;
struct i {
  template <typename, typename e> using f = decltype(e());
  template <typename a, typename e> static b<ad<f<a, e>>> k(int);
};
template <typename h> struct l : i {
  using c = decltype(k<int, h>(0));
};
template <typename j, typename h> struct g<j, h> : l<h>::c {};
template <typename... a> using ah = g<a...>::c;
class m;
class n {
  void o(m &) const;
};
template <typename = void> struct al;
template <typename a> struct al<a *> {
  void operator()(a *, a *) {
    if (__builtin_is_constant_evaluated())
      ;
  }
};
template <> struct al<> {
  template <typename a, typename e> void operator()(a p, e *p2) {
    al<ah<a, e *>>{}(p, p2);
  }
};
class q {
  void *aq;
  void *ar;
  template <class au> void r(au, int, long) {
    al a;
    a(aq, ar);
  }
};
template <typename> class s : q {
  int az;
  long ba;

public:
  void t() { r(this, az, ba); }
};
class m {
  s<int> bd;

public:
  void m_fn5() { bd.t(); }
};
void n::o(m &p) const { p.m_fn5(); }
