// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9   -analyzer-checker=core,alpha.core -verify -fblocks %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -analyzer-checker=core,alpha.core -verify -fblocks %s
// expected-no-diagnostics

//===------------------------------------------------------------------------------------------===//
// This files tests our path-sensitive handling of Objective-c++ files.
//===------------------------------------------------------------------------------------------===//

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// Test test1_aux() evaluates to char &.
char test1_as_rvalue() {
  return test1_aux();
}

// Test basic handling of references with Objective-C classes.
@interface Test1
- (char&) foo;
@end

char* Test1_harness(Test1 *p) {
  return &[p foo];
}

char Test1_harness_b(Test1 *p) {
  return [p foo];
}

// Basic test of C++ references with Objective-C pointers.
@interface RDar10569024
@property(readonly) int x;
@end

typedef RDar10569024* RDar10569024Ref;

void rdar10569024_aux(RDar10569024Ref o);

int rdar10569024(id p, id collection) {
  for (id elem in collection) {
    const RDar10569024Ref &o = (RDar10569024Ref) elem;
    rdar10569024_aux(o); // no-warning
    return o.x; // no-warning
  }
  return 0;
}
