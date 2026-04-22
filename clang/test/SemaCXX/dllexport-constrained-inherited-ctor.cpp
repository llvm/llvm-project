// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -fms-extensions -verify -std=c++20 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -fms-extensions -verify -std=c++20 -fno-dllexport-inlines %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -fsyntax-only -fms-extensions -verify -std=c++20 %s

// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/185924
// dllexport should not attempt to instantiate inherited constructors whose
// requires clause is not satisfied.
//
// This exercises two paths in checkClassLevelDLLAttribute:
//   1) findInheritingConstructor must skip constrained-out base ctors
//   2) dllexport propagated to the base template specialization must not
//      export members whose requires clause is not satisfied
//
// The constructor/method bodies are intentionally ill-formed when the
// constraint is not satisfied, so that forced instantiation via dllexport
// would produce an error without the correct fix.

template <bool B>
struct ConstrainedBase {
  struct Enabler {};
  ConstrainedBase(Enabler) requires(B) {}
  ConstrainedBase() requires(B) : ConstrainedBase(Enabler{}) {}
  ConstrainedBase(int);
};

struct __declspec(dllexport) ConstrainedChild : ConstrainedBase<false> {
  using ConstrainedBase::ConstrainedBase;
};

// Non-constructor constrained method on a base template specialization.
// When dllexport propagates to the base, methods whose requires clause
// is not satisfied must be skipped.
template <typename T>
struct BaseWithConstrainedMethod {
  void foo() requires(sizeof(T) > 100) { T::nonexistent(); }
  void bar() {}
};

struct __declspec(dllexport) MethodChild : BaseWithConstrainedMethod<int> {};
