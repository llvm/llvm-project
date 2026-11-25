// RUN: %clang_cc1 %s -verify -DDIAG1
// RUN: %clang_cc1 %s -verify -DDIAG1 -DDIAG2 -Wdelete-non-virtual-dtor
// RUN: %clang_cc1 %s -verify -DDIAG1         -Wmost -Wno-delete-non-abstract-non-virtual-dtor
// RUN: %clang_cc1 %s -verify         -DDIAG2 -Wmost -Wno-delete-abstract-non-virtual-dtor
// RUN: %clang_cc1 %s -verify                 -Wmost -Wno-delete-non-virtual-dtor

#ifndef DIAG1
#ifndef DIAG2
// expected-no-diagnostics
#endif
#endif

struct S1 {
  ~S1() {}
  virtual void abs() = 0;
};

void f1(S1 *s1) { delete s1; }
#ifdef DIAG1
// expected-warning@-2 {{delete called on 'S1' that is abstract but has non-virtual destructor}}
#endif

struct S2 {
  ~S2() {}
  virtual void real() {}
};
void f2(S2 *s2) { delete s2; }
#ifdef DIAG2
// expected-warning@-2 {{delete called on non-final 'S2' that has virtual functions but non-virtual destructor}}
#endif

namespace std {
struct destroying_delete_t {
  explicit destroying_delete_t() = default;
};
} // namespace std

// GH65524: a destroying operator delete takes over destruction, so the delete
// expression never calls the destructor and there is nothing to warn about.
struct S3 {
  virtual void abs() = 0;
  void operator delete(S3 *, std::destroying_delete_t);
};
void f3(S3 *s3) { delete s3; }

struct S4 {
  ~S4() {}
  virtual void abs() = 0;
  void operator delete(S4 *, std::destroying_delete_t);
};
void f4(S4 *s4) { delete s4; }

struct S5 : S3 {
  void abs() override;
};
void f5(S5 *s5) { delete s5; }

struct S6 {
  virtual void real() {}
  void operator delete(S6 *, std::destroying_delete_t);
};
void f6(S6 *s6) { delete s6; }

// The global operator delete and operator delete[] are never destroying, so
// the destructor is still invoked and the warning must stay.
void f7(S3 *s3) { ::delete s3; }
#ifdef DIAG1
// expected-warning@-2 {{delete called on 'S3' that is abstract but has non-virtual destructor}}
#endif
void f8(S3 *s3) { delete[] s3; }
#ifdef DIAG1
// expected-warning@-2 {{delete called on 'S3' that is abstract but has non-virtual destructor}}
#endif
