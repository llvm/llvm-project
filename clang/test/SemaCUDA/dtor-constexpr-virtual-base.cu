// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify=host
// RUN: %clang_cc1 %s -std=c++20 -fcuda-is-device -fsyntax-only -verify=dev

// host-no-diagnostics
// dev-no-diagnostics

#include "Inputs/cuda.h"

// The implicit destructor of an abstract class with virtual bases should
// consider those virtual bases during CUDA target inference, since the
// complete destructor variant destroys them.
void host_only();

constexpr void wraps_host() {
  if (!__builtin_is_constant_evaluated())
    host_only();
}

struct HasDtor {
  ~HasDtor() { wraps_host(); }
};

struct Base {
  HasDtor m;
  virtual ~Base();
};

template <class T>
struct Derived : virtual public Base {
  virtual void foo() = 0;
};

template class Derived<int>;
