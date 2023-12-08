// RUN: c-index-test -write-pch %t.h.pch -x objective-c %s-2.h
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_FAILONERROR=1 \
// RUN:   c-index-test -test-load-source-reparse 3 local %s -include %t.h
// RUN: c-index-test -write-pch %t.h.pch -x objective-c %s-3.h
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_FAILONERROR=1 \
// RUN:   c-index-test -test-load-source-reparse 3 local %s -include %t.h

#import "preamble-reparse-import.m-1.h"

void foo(void);
#import "preamble-reparse-import.m-2.h"
#import "preamble-reparse-import.m-1.h"
