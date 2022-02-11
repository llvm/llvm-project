// RUN: %clang_cc1 -std=c++1z %s -verify -triple x86_64-linux-gnu -DALIGN=16
// RUN: %clang_cc1 -std=c++1z %s -verify -fnew-alignment=2 -DALIGN=2
// RUN: %clang_cc1 -std=c++1z %s -verify -fnew-alignment=256 -DALIGN=256
// RUN: %clang_cc1 -std=c++1z %s -verify -triple wasm32-unknown-unknown -fnew-alignment=256 -DALIGN=256

// expected-no-diagnostics

#if ALIGN != __STDCPP_DEFAULT_NEW_ALIGNMENT__
#error wrong value for __STDCPP_DEFAULT_NEW_ALIGNMENT__
#endif
