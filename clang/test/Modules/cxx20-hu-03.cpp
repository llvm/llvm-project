// Test check that processing headers as C++20 units allows #pragma once.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-01.h \
// RUN: -Werror -o hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-02.h \
// RUN: -fmodule-file=%t/hu-01.pcm -o hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -fsyntax-only imports-01.cpp \
// RUN: -fmodule-file=%t/hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -fsyntax-only imports-02.cpp \
// RUN: -fmodule-file=%t/hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -fsyntax-only imports-03.cpp \
// RUN: -fmodule-file=%t/hu-02.pcm

//--- hu-01.h
#pragma once
struct HU {
  int a;
};
// expected-no-diagnostics

//--- hu-02.h
export import "hu-01.h";
// expected-no-diagnostics

//--- imports-01.cpp
import "hu-01.h";

HU foo(int x) {
  return {x};
}
// expected-no-diagnostics

//--- imports-02.cpp
import "hu-02.h";

HU foo(int x) {
  return {x};
}
// expected-no-diagnostics

//--- imports-03.cpp
import "hu-01.h";
import "hu-02.h";

HU foo(int x) {
  return {x};
}
// expected-no-diagnostics
