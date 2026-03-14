// Include a file to ensure we have multiple sources.
#include "sample.h"

// This introduces a symbol, a reference and a relation.
struct Bar : public Foo {
  // This introduces an OverriddenBy relation by implementing Foo::Func.
  void Func() override {}
};
