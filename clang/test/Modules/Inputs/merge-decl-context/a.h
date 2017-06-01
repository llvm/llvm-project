#ifndef A_H
#define A_H

template <typename T>
struct A {
  template <typename I>
  A(I i1, I i2) {
  }
  A(double) {}
  A(double, double) {}
  A(double, int) {}
  A(int, double) {}
};

template <typename T1, typename T2>
T1 fff(T2* t) {
  return T1(t, t);
}

inline A<int> ff(int i) {
  return fff<A<int>>(&i);
}

struct Aggregate {
  int member;
};
bool operator==(Aggregate, Aggregate) = delete;

#endif
