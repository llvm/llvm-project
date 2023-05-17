// RUN: %check_clang_tidy %s bugprone-suspicious-realloc-usage %t

void *realloc(void *, __SIZE_TYPE__);

namespace std {
  using ::realloc;
}

namespace n1 {
  void *p;
}

namespace n2 {
  void *p;
}

struct P {
  void *p;
  void *q;
  P *pp;
  void *&f();
};

struct P1 {
  static void *p;
  static void *q;
};

template<class>
struct P2 {
  static void *p;
  static void *q;
};

template<class A, class B>
void templ(void *p) {
  A::p = realloc(A::p, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'A::p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]
  p = realloc(A::p, 10);
  A::q = realloc(A::p, 10);
  A::p = realloc(B::p, 10);
  P2<A>::p = realloc(P2<A>::p, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'P2<A>::p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]
  P2<A>::p = realloc(P2<B>::p, 1);
}

void *&getPtr();
P &getP();

void warn(void *p, P *p1, int *pi) {
  p = realloc(p, 111);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]

  p = std::realloc(p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]

  p1->p = realloc(p1->p, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'p1->p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]

  p1->pp->p = realloc(p1->pp->p, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'p1->pp->p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]

  pi = (int*)realloc(pi, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'pi' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]

  templ<P1, P2<int>>(p);
}

void no_warn(void *p, P *p1, P *p2) {
  void *q = realloc(p, 10);
  q = realloc(p, 10);
  p1->q = realloc(p1->p, 10);
  p2->p = realloc(p1->p, 20);
  n1::p = realloc(n2::p, 30);
  p1->pp->p = realloc(p1->p, 10);
  getPtr() = realloc(getPtr(), 30);
  getP().p = realloc(getP().p, 20);
  p1->f() = realloc(p1->f(), 30);
}

void no_warn_if_copy_exists_before1(void *p) {
  void *q = p;
  p = realloc(p, 111);
}

void no_warn_if_copy_exists_before2(void *p, void *q) {
  q = p;
  p = realloc(p, 111);
}

void *g_p;

void no_warn_if_copy_exists_before3() {
  void *q = g_p;
  g_p = realloc(g_p, 111);
}

void warn_if_copy_exists_after(void *p) {
  p = realloc(p, 111);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]
  void *q = p;
}

void test_null_child(void *p) {
  for (;;)
    break;
  p = realloc(p, 111);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer [bugprone-suspicious-realloc-usage]
}
