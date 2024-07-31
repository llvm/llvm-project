// RUN: cp %s %t.cpp
// RUN: clang-include-cleaner -edit %t.cpp -- -I%S/Inputs/modules -fimplicit-module-maps -fmodules-strict-decluse -fmodule-name=XA
// RUN: FileCheck --match-full-lines --check-prefix=EDIT %s < %t.cpp

// Verify the tool still works on compilable-but-layering-violation code.
#include "a.h"
// EDIT-NOT: {{^}}#include "a.h"{{$}}
