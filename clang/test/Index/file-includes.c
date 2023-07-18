
#include "targeted-top.h"
#include "targeted-preamble.h"

extern int LocalVar;
int LocalVar;

// RUN: c-index-test -write-pch %t.h.pch %S/targeted-top.h -Xclang -detailed-preprocessing-record

// RUN: c-index-test -file-includes-in=%s %s | FileCheck %s -check-prefix=LOCAL
// RUN: env CINDEXTEST_EDITING=1 c-index-test -file-includes-in=%s %s | FileCheck %s -check-prefix=LOCAL
// RUN: c-index-test -file-includes-in=%s %s -include %t.h | FileCheck %s -check-prefix=LOCAL
// RUN: env CINDEXTEST_EDITING=1 c-index-test -file-includes-in=%s %s -include %t.h | FileCheck %s -check-prefix=LOCAL

// LOCAL: inclusion directive=targeted-top.h ({{.*[/\\]}}test{{[/\\]}}Index{{[/\\]}}targeted-top.h) {{.*}}=[2:1 - 2:2]
// LOCAL: inclusion directive=targeted-preamble.h ({{.*[/\\]}}test{{[/\\]}}Index{{[/\\]}}targeted-preamble.h) =[3:1 - 3:2]

// RUN: c-index-test -file-includes-in=%S/targeted-top.h %s | FileCheck %s -check-prefix=TOP
// RUN: env CINDEXTEST_EDITING=1 c-index-test -file-includes-in=%S/targeted-top.h %s | FileCheck %s -check-prefix=TOP
// RUN: c-index-test -file-includes-in=%S/targeted-top.h %s -include %t.h | FileCheck %s -check-prefix=TOP
// RUN: env CINDEXTEST_EDITING=1 c-index-test -file-includes-in=%S/targeted-top.h %s -include %t.h | FileCheck %s -check-prefix=TOP

// TOP: inclusion directive=targeted-nested1.h ({{.*[/\\]}}test{{[/\\]}}Index{{[/\\]}}targeted-nested1.h) =[5:1 - 5:2]
// TOP: inclusion directive=targeted-fields.h ({{.*[/\\]}}test{{[/\\]}}Index{{[/\\]}}targeted-fields.h) =[16:1 - 16:2]

// rdar://13803893
// RUN: c-index-test -file-includes-in=%S/Inputs/empty.h %S/Inputs/empty.h
