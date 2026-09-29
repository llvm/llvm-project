// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DOUTSIDE %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DINSIDE %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DNESTED %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DOUTSIDE -DEXTERN %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DINSIDE -DEXTERN %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fsyntax-only -verify -DNESTED -DEXTERN %s

// expected-no-diagnostics
#ifdef EXTERN
extern "C"
#endif
unsigned __int64 _umul128(unsigned __int64, unsigned __int64,
                          unsigned __int64 *);
namespace {
#ifdef INSIDE
  #pragma intrinsic(_umul128)
#endif
#ifdef NESTED
  namespace {
#pragma intrinsic(_umul128)
  }
#endif
}

#ifdef OUTSIDE
#pragma intrinsic(_umul128)
#endif

void foo() {
  unsigned __int64 carry;
  unsigned __int64 low = _umul128(0, 0, &carry);
}
