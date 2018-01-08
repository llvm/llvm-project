// RUN: %check_clang_tidy %s misc-suspicious-semicolon %t -- -- -fno-objc-arc -fobjc-abi-version=1

// This test ensures check_clang_tidy.py allows disabling Objective-C ARC and
// Objective-C 2.0 via passing arguments after -- on the command line.
//
// (We could include a test which doesn't pass any arguments after --
// to check if ARC and ObjC 2.0 are disabled by default, but that test
// could change behavior based on the default Objective-C runtime for
// the platform, which would make this test flaky.)

#if __has_feature(objc_arc)
#error Objective-C ARC unexpectedly enabled even with -fno-objc-arc
#endif

#ifdef __OBJC2__
#error Objective-C 2.0 unexpectedly enabled even with -fobjc-abi-version=1
#endif

@interface Foo
- (int)shouldDoStuff;
- (void)nop;
@end

void fail(Foo *f)
{
  if([f shouldDoStuff]); [f nop];
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: potentially unintended semicolon [misc-suspicious-semicolon]
  // CHECK-FIXES: if([f shouldDoStuff]) [f nop];
}
