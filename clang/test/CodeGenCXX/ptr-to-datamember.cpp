// RUN: %clang_cc1 -emit-llvm -o - %s

extern "C" int printf(...);

struct F {
  F() : iF(1), fF(2.0) {}
  int iF;
  float fF;
};

struct V {
  double d;
  int iV;
};

struct B  : virtual V{
  double d;
  int iB;
};

struct B1  : virtual V{
  double d;
  int iB1;
};

class A  : public B, public B1 {
public:
  A() : f(1.0), d(2.0), Ai(3) {}
  float f;
  double d;
  int Ai;
  F Af;
}; 

template <typename T> struct TT {
  int T::t::*pti;
};

struct I {
  typedef I t;
  int x;
};

void pr(const F& b) {
  printf(" %d %f\n", b.iF, b.fF);
}

void test_aggr_pdata(A& a1) {
  F A::* af = &A::Af;
  pr(a1.*af);

  (a1.*af).iF = 100;
  (a1.*af).fF = 200.00;
  printf(" %d %f\n", (a1.*af).iF, (a1.*af).fF);
  pr(a1.*af);

  (a1.*af).iF++;
  (a1.*af).fF--;
  --(a1.*af).fF;
  pr(a1.*af);
}

void test_aggr_pdata_1(A* pa) {
  F A::* af = &A::Af;
  pr(pa->*af);

  (pa->*af).iF = 100;
  (pa->*af).fF = 200.00;
  printf(" %d %f\n", (pa->*af).iF, (pa->*af).fF);
  pr(pa->*af);

  (pa->*af).iF++;
  (pa->*af).fF--;
  --(pa->*af).fF;
  pr(pa->*af);
}

int main() 
{
  A a1;
  TT<I> tt;
  I i;
  int A::* pa = &A::Ai;
  float A::* pf = &A::f;
  double A::* pd = &A::d;
  tt.pti = &I::x;
  printf("%d %d %d\n", &A::Ai, &A::f, &A::d);
  printf("%d\n", &A::B::iB);
  printf("%d\n", &A::B1::iB1);
  printf("%d\n", &A::f);
  printf("%d\n", &A::B::iV);
  printf("%d\n", &A::B1::iV);
  printf("%d\n", &A::B::V::iV);
  printf("%d\n", &A::B1::V::iV);
  printf("%d, %f, %f  \n", a1.*pa, a1.*pf, a1.*pd);
  printf("%d\n", i.*tt.pti);
  test_aggr_pdata(a1);
  test_aggr_pdata_1(&a1);
}
