// RUN: %check_clang_tidy -std=c++11,c++14 %s modernize-use-noexcept %t -- -- -fexceptions
// This test is not run in C++17 or later because dynamic exception
// specifications were removed in C++17.

class A {};
class B {};

void foo() throw();
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: void foo() noexcept;

template <typename T>
void foo() throw();
void footest() { foo<int>(); foo<double>(); }
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: void foo() noexcept;

void bar() throw(...);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: dynamic exception specification 'throw(...)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void bar() noexcept(false);

void k() throw(int(int));
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: dynamic exception specification 'throw(int(int))' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void k() noexcept(false);

void foobar() throw(A, B)
{}
// CHECK-MESSAGES: :[[@LINE-2]]:15: warning: dynamic exception specification 'throw(A, B)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void foobar() noexcept(false)

void baz(int = (throw A(), 0)) throw(A, B) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: dynamic exception specification 'throw(A, B)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void baz(int = (throw A(), 0)) noexcept(false) {}

void g(void (*fp)(void) throw());
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: void g(void (*fp)(void) noexcept);

void f(void (*fp)(void) throw(int)) throw(char);
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: dynamic exception specification 'throw(int)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: dynamic exception specification 'throw(char)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void f(void (*fp)(void) noexcept(false)) noexcept(false);

#define THROW throw
void h(void (*fp)(void) THROW(int)) THROW(char);
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: dynamic exception specification 'THROW(int)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: dynamic exception specification 'THROW(char)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void h(void (*fp)(void) noexcept(false)) noexcept(false);

void j() throw(int(int) throw(void(void) throw(int)));
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: dynamic exception specification 'throw(int(int) throw(void(void) throw(int)))' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void j() noexcept(false);

class Y {
  Y() throw() = default;
};
// CHECK-MESSAGES: :[[@LINE-2]]:7: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: Y() noexcept = default;

struct Z {
  void operator delete(void *ptr) throw();
  void operator delete[](void *ptr) throw(int);
  ~Z() throw(int) {}
};
// CHECK-MESSAGES: :[[@LINE-4]]:35: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-4]]:37: warning: dynamic exception specification 'throw(int)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-4]]:8: warning: dynamic exception specification 'throw(int)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]
// CHECK-FIXES: void operator delete(void *ptr) noexcept;
// CHECK-FIXES: void operator delete[](void *ptr) noexcept(false);
// CHECK-FIXES: ~Z() noexcept(false) {}

struct S {
  void f() throw();
};
void f(void (S::*)() throw());
// CHECK-MESSAGES: :[[@LINE-3]]:12: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-2]]:22: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: void f() noexcept;
// CHECK-FIXES: void f(void (S::*)() noexcept);

template <typename T>
struct ST {
  void foo() throw();
};
template <typename T>
void ft(void (ST<T>::*)() throw());
// CHECK-MESSAGES: :[[@LINE-4]]:14: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-2]]:27: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: void foo() noexcept;
// CHECK-FIXES: void ft(void (ST<T>::*)() noexcept);

typedef void (*fp)(void (*fp2)(int) throw());
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: dynamic exception specification 'throw()' is deprecated; consider using 'noexcept' instead [modernize-use-noexcept]
// CHECK-FIXES: typedef void (*fp)(void (*fp2)(int) noexcept);

// Should not trigger a replacement.
void titi() noexcept {}
void toto() noexcept(true) {}

// Should not trigger a replacement.
void bad()
#if !__has_feature(cxx_noexcept)
    throw()
#endif
  ;
