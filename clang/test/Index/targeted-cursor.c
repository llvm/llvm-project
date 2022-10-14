
#include "targeted-top.h"
#include "targeted-preamble.h"

int LocalVar1;
int LocalVar2;

// RUN: c-index-test -write-pch %t.h.pch %S/targeted-top.h
// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -cursor-at=%s:5:10 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -cursor-at=%S/targeted-top.h:11:15 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=vector_get_x  \
// RUN:  | FileCheck %s -check-prefix=TOP-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -cursor-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -cursor-at=%S/targeted-fields.h:2:7 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=FIELD-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -cursor-at=%S/targeted-fields.h:1:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=FIELD-CURSOR2

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -cursor-at=%s:5:10 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -cursor-at=%S/targeted-top.h:11:15 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=vector_get_x  \
// RUN:  | FileCheck %s -check-prefix=TOP-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -cursor-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED-CURSOR1

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -cursor-at=%S/targeted-preamble.h:2:15 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=PREAMBLE-CURSOR1

// LOCAL-CURSOR1: VarDecl=LocalVar1:5:5
// TOP-CURSOR1: VarDecl=TopVar:11:12
// NESTED-CURSOR1: VarDecl=NestedVar1:2:12
// PREAMBLE-CURSOR1: VarDecl=PreambleVar:2:12

// FIELD-CURSOR1: FieldDecl=z:2:7 (Definition)
// FIELD-CURSOR2: StructDecl=Vector:13:9 (Definition)
