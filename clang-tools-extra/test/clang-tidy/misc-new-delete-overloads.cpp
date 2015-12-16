// RUN: %check_clang_tidy %s misc-new-delete-overloads %t -- -- -std=c++14

typedef decltype(sizeof(int)) size_t;

struct S {
  // CHECK-MESSAGES: :[[@LINE+1]]:9: warning: declaration of 'operator new' has no matching declaration of 'operator delete' at the same scope [misc-new-delete-overloads]
  void *operator new(size_t size) noexcept;
  // CHECK-MESSAGES: :[[@LINE+1]]:9: warning: declaration of 'operator new[]' has no matching declaration of 'operator delete[]' at the same scope
  void *operator new[](size_t size) noexcept;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: declaration of 'operator new' has no matching declaration of 'operator delete' at the same scope
void *operator new(size_t size) noexcept(false);

struct T {
  // Sized deallocations are not enabled by default, and so this new/delete pair
  // does not match. However, we expect only one warning, for the new, because
  // the operator delete is a placement delete and we do not warn on mismatching
  // placement operations.
  // CHECK-MESSAGES: :[[@LINE+1]]:9: warning: declaration of 'operator new' has no matching declaration of 'operator delete' at the same scope
  void *operator new(size_t size) noexcept;
  void operator delete(void *ptr, size_t) noexcept; // ok only if sized deallocation is enabled
};

struct U {
  void *operator new(size_t size) noexcept;
  void operator delete(void *ptr) noexcept;

  void *operator new[](size_t) noexcept;
  void operator delete[](void *) noexcept;
};

struct Z {
  // CHECK-MESSAGES: :[[@LINE+1]]:8: warning: declaration of 'operator delete' has no matching declaration of 'operator new' at the same scope
  void operator delete(void *ptr) noexcept;
  // CHECK-MESSAGES: :[[@LINE+1]]:8: warning: declaration of 'operator delete[]' has no matching declaration of 'operator new[]' at the same scope
  void operator delete[](void *ptr) noexcept;
};

struct A {
  void *operator new(size_t size, Z) noexcept; // ok, placement new
};

struct B {
  void operator delete(void *ptr, A) noexcept; // ok, placement delete
};

// It is okay to have a class with an inaccessible free store operator.
struct C {
  void *operator new(size_t, A) noexcept; // ok, placement new
private:
  void operator delete(void *) noexcept;
};

// It is also okay to have a class with a delete free store operator.
struct D {
  void *operator new(size_t, A) noexcept; // ok, placement new
  void operator delete(void *) noexcept = delete;
};

struct E : U {
  void *operator new(size_t) noexcept; // okay, we inherit operator delete from U
};

struct F : S {
  // CHECK-MESSAGES: :[[@LINE+1]]:9: warning: declaration of 'operator new' has no matching declaration of 'operator delete' at the same scope
  void *operator new(size_t) noexcept;
};

class G {
  void operator delete(void *) noexcept;
};

struct H : G {
  // CHECK-MESSAGES: :[[@LINE+1]]:9: warning: declaration of 'operator new' has no matching declaration of 'operator delete' at the same scope
  void *operator new(size_t) noexcept; // base class operator is inaccessible
};

template <typename Base> struct Derived : Base {
  void operator delete(void *);
};
