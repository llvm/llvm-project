
// RUN: %clang_cc1 -triple wasm32 -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
// RUN: %clang_cc1 -triple thumbv7 -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
// RUN: %clang_cc1 -DEXPOK -triple aarch64 -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
// RUN: %clang_cc1 -DEXPOK -triple x86_64 -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
// RUN: %clang_cc1 -DEXPOK -triple systemz -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
// RUN: %clang_cc1 -DEXPOK -triple powerpc -fsyntax-only -Wno-unknown-pragmas -Wignored-pragmas -verify %s
#ifdef EXPOK
// expected-no-diagnostics
#else
// expected-warning@+4 {{'#pragma float_control' is not supported on this target - ignored}}
// expected-warning@+5 {{'#pragma FENV_ACCESS' is not supported on this target - ignored}}
// expected-warning@+6 {{'#pragma FENV_ROUND' is not supported on this target - ignored}}
#endif
#pragma float_control(precise, on)

#pragma STDC FENV_ACCESS OFF

#pragma STDC FENV_ROUND FE_DOWNWARD
