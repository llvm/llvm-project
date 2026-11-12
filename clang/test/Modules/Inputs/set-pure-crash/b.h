#pragma once

#include "a.h"
#include "c.h"

template <typename Fun, typename = simple<Fun>>
void foo(Fun) {}

class Child : public Base<Tag> {
public:
  void func() {
    foo([]() {});
  }
};
