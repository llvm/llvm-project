// RUN: split-file %s %t.dir
// RUN: %clang_cc1 -verify %t.dir/defined.cpp
// RUN: %clang_cc1 -verify -mthread-model posix %t.dir/defined.cpp
// RUN: %clang_cc1 -verify -mthread-model single %t.dir/not-defined.cpp
// RUN: %clang_cc1 -verify -x c %t.dir/not-defined.cpp

//--- defined.cpp
// expected-no-diagnostics
#ifndef __STDCPP_THREADS__
#error __STDCPP_THREADS__ is not defined in posix thread model.
#endif

//--- not-defined.cpp
// expected-no-diagnostics
#ifdef __STDCPP_THREADS__
#error __STDCPP_THREADS__ is defined in single thread model.
#endif
