
#include "targeted-top.h"

// This tests that we will correctly error out on the deserialized decl.

// RUN: c-index-test -write-pch %t.h.pch %S/targeted-top.h
// RUN: env CINDEXTEST_FAILONERROR=1 not c-index-test -cursor-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1
// RUN: env CINDEXTEST_FAILONERROR=1 not c-index-test -cursor-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1 2>&1 \
// RUN:  | FileCheck %s

// CHECK: error: 'NestedVar1' was deserialized
