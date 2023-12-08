@interface Foo
- (void) test;
- (void) test2;
@end

@implementation Foo
- (void) test {
  [_self test2];
}
- (void) test2 {}
@end

// RUN: rm -f %t
// RUN: not %clang -Wall -fsyntax-only %s --serialize-diagnostics %t.diag > /dev/null 2>&1
// RUN: c-index-test -read-diagnostics %t.diag > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// This test checks that serialized diagnostics handle notes with no source location.

// CHECK: {{.*[/\\]}}serialized-diags.m:8:4: error: use of undeclared identifier '_self'; did you mean 'self'? [] [Semantic Issue]
// CHECK: Range: {{.*[/\\]}}serialized-diags.m:8:4 {{.*[/\\]}}serialized-diags.m:8:9
// CHECK: Number FIXITs = 1
// CHECK: FIXIT: ({{.*[/\\]}}serialized-diags.m:8:4 - {{.*[/\\]}}serialized-diags.m:8:9): "self"
// CHECK: +-(null):0:0: note: 'self' is an implicit parameter [] [Semantic Issue]
// CHECK: Number FIXITs = 0
// CHECK: {{.*[/\\]}}serialized-diags.m:1:12: warning: class 'Foo' defined without specifying a base class [-Wobjc-root-class] [Semantic Issue]
// CHECK: Number FIXITs = 0
// CHECK: +-{{.*[/\\]}}serialized-diags.m:1:15: note: add a super class to fix this problem [] [Semantic Issue]
// CHECK: Number FIXITs = 0
// CHECK: Number of diagnostics: 2
