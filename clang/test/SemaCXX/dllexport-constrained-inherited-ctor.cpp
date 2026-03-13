// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -fms-extensions -verify -std=c++20 %s
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

template <bool B>
struct ConstrainedBase {
  ConstrainedBase() requires(!B) = delete;
  ConstrainedBase() requires(B) {}
  ConstrainedBase(int);
};

struct __declspec(dllexport) ConstrainedChild : ConstrainedBase<false> {
  using ConstrainedBase::ConstrainedBase;
};
