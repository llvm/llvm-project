// Verify that ThreadSanitizer detects data races on Objective-C synthesized
// (nonatomic, retain) and (nonatomic, copy) properties under manual retain
// release. The synthesized setters call objc_setProperty_nonatomic and
// objc_setProperty_nonatomic_copy, which store into the ivar inside libobjc.
// Also verify that races on a property whose setter lives in a
// non-instrumented module are not reported.

// RUN: basename %t-lib.dylib | tr -d '\n' > %t.basename
// RUN: %clangxx_tsan -shared %s -fno-sanitize=thread -fno-objc-arc -DBUILD_LIB \
// RUN:   -framework Foundation -o %t-lib.dylib \
// RUN:   -install_name @rpath/%{readfile:%t.basename}
// RUN: %clangxx_tsan -fno-objc-arc %s %t-lib.dylib -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

@interface UninstrumentedContainer : NSObject
@property(nonatomic, retain) NSString *text;
@end

#ifdef BUILD_LIB

@implementation UninstrumentedContainer
@end

#else

#  import "../test.h"

@interface MRCContainer : NSObject
@property(nonatomic, retain) NSString *retained;
@property(nonatomic, copy) NSString *copied;
// TSan reports only one race per address, so the read-vs-write cases use their
// own properties.
@property(nonatomic, retain) NSString *readRetained;
@property(nonatomic, copy) NSString *readCopied;
@end

@implementation MRCContainer
@end

static void RaceOn(void (^first)(void), void (^second)(void)) {
  dispatch_queue_t q =
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  barrier_init(&barrier, 2);
  dispatch_group_t group = dispatch_group_create();
  dispatch_group_async(group, q, ^{
    first();
    barrier_wait(&barrier);
  });
  dispatch_group_async(group, q, ^{
    barrier_wait(&barrier);
    second();
  });
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
  dispatch_release(group);
}

int main() {
  @autoreleasepool {
    MRCContainer *c = [MRCContainer new];
    UninstrumentedContainer *u = [UninstrumentedContainer new];
    NSString *s1 = [NSString stringWithFormat:@"hello-non-tagged-string-%d", 1];
    NSString *s2 = [NSString stringWithFormat:@"world-non-tagged-string-%d", 2];

    // 1. A setter in a non-instrumented module is not reported.
    RaceOn(
        ^{
          u.text = s1;
        },
        ^{
          u.text = s2;
        });
    fprintf(stderr, "Uninstrumented writes completed.\n");
    // CHECK-NOT: WARNING: ThreadSanitizer
    // CHECK: Uninstrumented writes completed.

    // 2. Write-vs-write race on a (nonatomic, retain) property.
    RaceOn(
        ^{
          c.retained = s1;
        },
        ^{
          c.retained = s2;
        });
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic{{ }}
    // CHECK-NEXT:     #1 -[MRCContainer setRetained:]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic{{ }}
    // CHECK-NEXT:     #1 -[MRCContainer setRetained:]

    // 3. Write-vs-write race on a (nonatomic, copy) property.
    RaceOn(
        ^{
          c.copied = s1;
        },
        ^{
          c.copied = s2;
        });
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[MRCContainer setCopied:]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[MRCContainer setCopied:]

    // 4. Read-vs-write race (getter vs setter) on a (nonatomic, retain)
    //    property.
    RaceOn(
        ^{
          c.readRetained = s1;
        },
        ^{
          (void)c.readRetained;
        });
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Read of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 -[MRCContainer readRetained]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic{{ }}
    // CHECK-NEXT:     #1 -[MRCContainer setReadRetained:]

    // 5. Read-vs-write race (getter vs setter) on a (nonatomic, copy) property.
    RaceOn(
        ^{
          c.readCopied = s1;
        },
        ^{
          (void)c.readCopied;
        });
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Read of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 -[MRCContainer readCopied]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[MRCContainer setReadCopied:]

    [c release];
    [u release];
  }
  fprintf(stderr, "Done.\n");
  // CHECK: Done.
  return 0;
}

#endif
