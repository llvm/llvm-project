struct {

@implementation Foo

- (void)finalize {
  NSLog(@"bar");
}

- (NSArray *)graphics {
}

@end

// Test that we don't crash
// RUN: c-index-test -test-load-source-reparse 3 local %s
