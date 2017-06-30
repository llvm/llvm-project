#include "Inputs/HeaderWithSymbol.h"

int main() {
  return 0; // CHECK: {{^  return 0;}}
}

// Test 1.
// The file IncludeHeaderWithSymbol.cpp doesn't contain the symbol Foo
// and is expected to be written to stdout without modifications
// RUN: clang-rename -qualified-name=Foo -new-name=Bar %s -- | FileCheck %s
