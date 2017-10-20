// RUN: %check_clang_tidy %s misc-suspicious-semicolon %t

// This test checks if Objective-C 2.0 (@properties) and
// Automatic Reference Counting (ARC) are enabled for .m files
// checked via check_clang_tidy.py.

#if !__has_feature(objc_arc)
#error Objective-C ARC not enabled as expected
#endif

@interface Foo
@property (nonatomic, assign) int shouldDoStuff;
- (void)nop;
@end

void fail(Foo *f)
{
  if(f.shouldDoStuff); [f nop];
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: potentially unintended semicolon [misc-suspicious-semicolon]
  // CHECK-FIXES: if(f.shouldDoStuff) [f nop];
}
