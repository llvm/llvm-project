// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -std=c++11 -Wno-return-stack-address -Wreturn-local-addr %s

namespace PR26599 {
template <typename>
struct S;

struct I {};

template <typename T>
void *&non_pointer() {
  void *&r = S<T>()[I{}];
  return r;
}

template <typename T>
void *&pointer() {
  void *&r = S<T>()[nullptr];
  return r;
}
}

namespace LocalTemporary {

template <class T>
class QMap {
public:
  T value(const T &t = T()) const {
    return t;
  }
};

struct A {};

void test() {
  QMap<A *> map;
  map.value();
}

typedef int* ptr;
ptr int1(const ptr &p = ptr()) {
  return (p);
}

ptr int2(const ptr &p = nullptr) {
  return p;
}

ptr int3() {
  const ptr &p = ptr();
  return p;
}

const int *int4(const int &x = 5) {
  return &x;
}

const int *int5(const int &x) {
  return &x;
}

const int *int6() {
  const int &x = 11;  //expected-note{{binding reference variable 'x' here}}
  return &x;  //expected-warning{{returning address of local temporary object}}
}

const int *int7(int x) {
  const int &x2 = x;  // expected-note{{binding reference variable 'x2' here}}
  return &x2;  //  expected-warning{{address of stack memory associated with parameter 'x' returned}}
}

const int *int8(const int &x = 5) {
  const int &x2 = x;
  return &x2;
}

const int *int9() {
  const int &x = 5;  // expected-note{{binding reference variable 'x' here}}
  const int &x2 = x;  // expected-note{{binding reference variable 'x2' here}}
  return &x2;  // expected-warning{{returning address of local temporary object}}
}
}
