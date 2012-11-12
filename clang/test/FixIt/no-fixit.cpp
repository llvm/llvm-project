// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -x c++ -std=c++11 %s 2>&1 | FileCheck %s

// test that the diagnostics produced by this code do not include fixit hints

// CHECK-NOT: fix-it:

template<template<typename> +> void func();

struct {
  void i() {
    (void)&i;
  }
} x;
