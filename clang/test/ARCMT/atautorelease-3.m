// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

@interface NSAutoreleasePool
- drain;
+new;
+alloc;
-init;
-autorelease;
- release;
@end

void NSLog(id, ...);

void test1(int x) {
  // All this stuff get removed since nothing is happening inside.
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSAutoreleasePool *chunkPool = [[NSAutoreleasePool alloc] init];
  while (x) {
    chunkPool = [[NSAutoreleasePool alloc] init];
    [chunkPool release];
  }

  [chunkPool drain];
  [pool drain];
}

void test2(int x) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSAutoreleasePool *chunkPool = [[NSAutoreleasePool alloc] init];
  while (x) {
    chunkPool = [[NSAutoreleasePool alloc] init];
    ++x;
    [chunkPool release];
  }

  [chunkPool drain];
  [pool drain];
}
