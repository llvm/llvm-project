// RUN: %clang_cc1 -triple x86_64-linux-gnu -fms-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions -fsyntax-only -verify %s
// expected-no-diagnostics

// intrin0.h and xsaveintrin.h declare these intrinsics with __int64, which is
// 'long long' on every target. The builtin types must match: int64_t is 'long'
// on LP64 targets such as x86_64-linux-gnu, where a mismatch makes each
// declaration below a "conflicting types" error.

__int64 _InterlockedAnd64(__int64 volatile *, __int64);
__int64 _InterlockedDecrement64(__int64 volatile *);
__int64 _InterlockedExchange64(__int64 volatile *, __int64);
__int64 _InterlockedExchangeAdd64(__int64 volatile *, __int64);
__int64 _InterlockedExchangeSub64(__int64 volatile *, __int64);
__int64 _InterlockedIncrement64(__int64 volatile *);
__int64 _InterlockedOr64(__int64 volatile *, __int64);
__int64 _InterlockedXor64(__int64 volatile *, __int64);
unsigned __int64 _xgetbv(unsigned int);
void _xsetbv(unsigned int, unsigned __int64);
