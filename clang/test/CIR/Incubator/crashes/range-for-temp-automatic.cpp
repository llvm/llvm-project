// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// SD_Automatic storage duration for reference temporaries in range-based for loops
// Location: CIRGenExpr.cpp:2435

template <typename> struct b;
template <typename> struct f;
template <typename c> struct f<c *> {
  typedef c d;
};
template <typename e, typename> class j {
public:
  f<e>::d operator*();
  void operator++();
};
template <typename e, typename g> bool operator!=(j<e, g>, j<e, g>);
template <typename> class k;
template <typename c> struct b<k<c>> {
  using h = c *;
};
template <typename i> struct F {
  typedef b<i>::h h;
  ~F();
};
template <typename c, typename i = k<c>> class G : F<i> {
public:
  typedef j<typename F<i>::h, int> iterator;
  iterator begin();
  iterator end();
};
template <typename l> class m {
public:
  using n = l;
  using o = n *;
  using iterator = o;
  iterator begin();
  iterator end();
};
class p {
public:
  G<p *> u();
  m<p *> r();
} q;
void s() {
  m a = q.r();
  for (p *v : a)
    for (p *t : v->u())
      ;
}
