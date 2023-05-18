// RUN: clang-include-cleaner -html=- %s -- -I %S/Inputs | FileCheck %s
#include "bar.h"
#include "foo.h"

int n = foo();
// CHECK: <span class='ref sel' data-hover='t{{[0-9]+}}'>foo</span>()
