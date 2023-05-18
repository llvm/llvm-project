
#include "targeted-top.h"
#include "targeted-preamble.h"

extern int LocalVar;
int LocalVar;

// RUN: c-index-test -write-pch %t.h.pch %S/targeted-top.h -Xclang -detailed-preprocessing-record
// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -file-refs-at=%s:5:17 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -file-refs-at=%s:5:17 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL

// LOCAL: VarDecl=LocalVar:5:12
// LOCAL: VarDecl=LocalVar:5:12 =[5:12 - 5:20]
// LOCAL: VarDecl=LocalVar:6:5 =[6:5 - 6:13]

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -file-refs-at=%S/targeted-top.h:14:7 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:  | FileCheck %s -check-prefix=TOP

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -file-refs-at=%S/targeted-top.h:14:7 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:  | FileCheck %s -check-prefix=TOP

// TOP: FieldDecl=x:14:7 (Definition)
// TOP: FieldDecl=x:14:7 (Definition) =[14:7 - 14:8]
// TOP: MemberRefExpr=x:14:7 SingleRefName=[20:13 - 20:14] RefName=[20:13 - 20:14] =[20:13 - 20:14]

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -file-refs-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -file-refs-at=%S/targeted-nested1.h:2:16 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED

// NESTED: VarDecl=NestedVar1:2:12
// NESTED: VarDecl=NestedVar1:2:12 =[2:12 - 2:22]

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -file-refs-at=%S/targeted-preamble.h:2:15 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=PREAMBLE

// PREAMBLE: VarDecl=PreambleVar:2:12
// PREAMBLE: VarDecl=PreambleVar:2:12 =[2:12 - 2:23]
