// RUN: c-index-test -write-pch %t.h.pch %s
// RUN: c-index-test -code-completion-at=%s:19:1 %s -include %t.h | FileCheck %s

// <rdar://12316296> clang Code Completion returns nothing but preprocessor macros

#ifndef HEADER
#define HEADER

@interface I
@end

// CHECK: FunctionDecl:{ResultType void}{TypedText foo}
void foo();

#else

@implementation I
-(void)meth {

}
@end

#endif
