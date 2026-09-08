// RUN: not %clang_cc1 -fherbceptions -fcxx-exceptions -fsyntax-only %s 2>&1 | FileCheck %s

// The herbception 'throws'/'return_failure{...}' specifier is part of the canonical
// function type because it changes the calling convention ({T, i1} return
// instead of T). Therefore:
//   - a throws function pointer is only compatible with a throws function,
//   - a plain or noexcept function cannot be used where a throws pointer is
//     expected, and vice versa,
//   - a virtual function override must use the same herbception specifier.

using size_t = __SIZE_TYPE__;
namespace std { struct error { void *domain; __SIZE_TYPE__ code; }; }

void throws_fn(size_t) throws {}
void plain_fn(size_t) {}
void noexcept_fn(size_t) noexcept {}

// CHECK: error: cannot initialize a variable of type 'void (*)(size_t)'{{.*}}with an rvalue of type 'void (*)(size_t) throws'
// CHECK: error: cannot initialize a variable of type 'void (*)(size_t) throws'{{.*}}with an lvalue of type 'void (size_t)'
// CHECK: error: cannot initialize a variable of type 'void (*)(size_t) throws'{{.*}}with an lvalue of type 'void (size_t) noexcept'
void fnptr_test() {
  void (*bad1)(size_t) = &throws_fn;          // throws -> plain: error
  void (*ok)(size_t) throws = &throws_fn;     // throws -> throws: ok
  void (*bad2)(size_t) throws = plain_fn;     // plain -> throws: error
  void (*bad3)(size_t) throws = noexcept_fn;  // noexcept -> throws: error
  (void)bad1; (void)ok; (void)bad2; (void)bad3;
}

struct BaseThrows {
  virtual void f() throws;
};
struct DerivedMissingThrows : BaseThrows {
  // CHECK: error: overriding function has a different herbceptions ('throws'/'return_failure{...}') specifier than the base version
  void f() override;
};

struct BasePlain {
  virtual void f();
};
struct DerivedAddsThrows : BasePlain {
  // CHECK: error: overriding function has a different herbceptions ('throws'/'return_failure{...}') specifier than the base version
  void f() throws override;
};

// Matching specs are fine.
struct DerivedMatchesThrows : BaseThrows {
  void f() throws override;
};
