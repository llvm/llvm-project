// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -I. m_template.cppm -emit-reduced-module-interface \
// RUN:            -o m_template.pcm
//
// RUN: %clang_cc1 -std=c++20 -I. m_wrapper.cppm \
// RUN:            -emit-reduced-module-interface \
// RUN:            -fmodule-file=m_template=m_template.pcm \
// RUN:            -o m_wrapper.pcm
//
// RUN: %clang_cc1 -std=c++20 -I. main.cpp \
// RUN:            -fmodule-file=m_template=m_template.pcm \
// RUN:            -fmodule-file=m_wrapper=m_wrapper.pcm \
// RUN:            -fsyntax-only -verify

//--- local_class.h
#pragma once

template <typename U>
auto make(U u) {
  static int counter;
  struct Box {
    U value;
    int id() const { return counter; }  // references the enclosing static local
    U unwrap() const { return value; }
  };
  return Box{u};
}

template <typename U>
auto trigger_ast_deserialization(U u) {
  return []() {};
}

//--- m_template.cppm
module;
#include "local_class.h"
export module m_template;
export using ::make;
export using ::trigger_ast_deserialization;

//--- m_wrapper.cppm
module;
#include "local_class.h"
export module m_wrapper;
import m_template;

export inline void TriggerInstantiation() {
  make(0);
  trigger_ast_deserialization(0);
}

//--- main.cpp
// expected-no-diagnostics
import m_template;
import m_wrapper;
#include "local_class.h"

struct Token {};

int main() {
  TriggerInstantiation();
  trigger_ast_deserialization(Token{});
  auto b = make(Token{});
  (void)b.unwrap();
  (void)b.id();
}
