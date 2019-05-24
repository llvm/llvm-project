// RUN: %check_clang_tidy -std=c++11,c++14 %s modernize-use-noexcept %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-noexcept.ReplacementString, value: 'NOEXCEPT'}]}" \
// RUN:   -- -fexceptions
// This test is not run in C++17 or later because dynamic exception
// specifications were removed in C++17.

// Example definition of NOEXCEPT -- simplified test to see if noexcept is supported.
#if (__has_feature(cxx_noexcept))
#define NOEXCEPT noexcept
#else
#define NOEXCEPT throw()
#endif

void bar() throw() {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: dynamic exception specification 'throw()' is deprecated; consider using 'NOEXCEPT' instead [modernize-use-noexcept]
// CHECK-FIXES: void bar() NOEXCEPT {}

// Should not trigger a FixItHint, since macros only support noexcept, and this
// case throws.
class A {};
class B {};
void foobar() throw(A, B);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: dynamic exception specification 'throw(A, B)' is deprecated; consider removing it instead [modernize-use-noexcept]

// Should not trigger a replacement.
void foo() noexcept(true);

struct Z {
  void operator delete(void *ptr) throw();
  void operator delete[](void *ptr) throw(int);
  ~Z() throw(int) {}
};
// CHECK-MESSAGES: :[[@LINE-4]]:35: warning: dynamic exception specification 'throw()' is deprecated; consider using 'NOEXCEPT' instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-4]]:37: warning: dynamic exception specification 'throw(int)' is deprecated; consider removing it instead [modernize-use-noexcept]
// CHECK-MESSAGES: :[[@LINE-4]]:8: warning: dynamic exception specification 'throw(int)' is deprecated; consider removing it instead [modernize-use-noexcept]
// CHECK-FIXES: void operator delete(void *ptr) NOEXCEPT;
// CHECK-FIXES: void operator delete[](void *ptr) throw(int);
// CHECK-FIXES: ~Z() throw(int) {}
