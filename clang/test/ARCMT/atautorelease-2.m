// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

@interface NSAutoreleasePool
- drain;
+new;
+alloc;
-init;
-autorelease;
-release;
@end

void NSLog(id, ...);

int main (int argc, const char * argv[]) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    NSAutoreleasePool *chunkPool = [[NSAutoreleasePool alloc] init];

    while (argc) {
      [chunkPool release];
      return 0;
    }

    [chunkPool drain];
    [pool drain];

    return 0;
}
