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

//--- template_class.h
#pragma once

class TemplateClass {
 public:
  template <typename U>
  auto f1(U u);
};

template <typename U>
auto TemplateClass::f1(U u) {
  U z = u;
  return [=](auto) {
    (void)z;
  };
}

template <typename U>
auto trigger_ast_deserialization(U u) {
  return []() {};
}

//--- m_template.cppm
module;
#include "template_class.h"
export module m_template;
export using ::TemplateClass;
export using ::trigger_ast_deserialization;

//--- m_wrapper.cppm
module;
#include "template_class.h"
export module m_wrapper;
import m_template;

export inline void TriggerInstantiation() {
  TemplateClass tc;
  tc.f1(0);
  trigger_ast_deserialization(0);
}

//--- main.cpp
// expected-no-diagnostics
import m_template;
import m_wrapper;
#include "template_class.h"

struct Token {};

int main() {
  TriggerInstantiation();
  trigger_ast_deserialization(Token{});
  TemplateClass tc;
  tc.f1(Token{});
}
