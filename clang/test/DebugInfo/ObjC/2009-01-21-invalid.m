// RUN: %clang_cc1 -S -debug-info-kind=limited -o %t.s %s

// FIXME: This test case can be removed at some point (since it will
// no longer effectively test anything). The reason it was causing
// trouble was the synthesized self decl in im1 was causing the debug
// info for I1* to be generated, but referring to an invalid compile
// unit. This was later referred to by f1 and created ill formed debug
// information.

@interface I1 @end

@implementation I1
-im0 { return 0; }
@end

I1 *f1(void) { return 0; }
