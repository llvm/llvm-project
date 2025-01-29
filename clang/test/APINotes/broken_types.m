// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s 2> %t.err
// RUN: FileCheck %s < %t.err

#include "BrokenTypes.h"

// CHECK: <API Notes>:1:1: error: unknown type name 'not_a_type'
// CHECK-NEXT: not_a_type
// CHECK-NEXT: ^

// CHECK: <API Notes>:1:7: error: unparsed tokens following type
// CHECK-NEXT: int * with extra junk
// CHECK-NEXT:       ^

// CHECK: BrokenTypes.h:4:6: error: API notes replacement type 'int *' has a different size from original type 'char'

// CHECK: BrokenTypes.h:6:13: error: API notes replacement type 'double' has a different size from original type 'char'

// CHECK: 5 errors generated.
