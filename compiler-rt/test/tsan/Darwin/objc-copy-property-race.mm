// Verify that ThreadSanitizer detects data races on Objective-C synthesized
// (nonatomic, copy) properties at both -O0 and -O2, and does not report races
// on (atomic, copy) properties.

// RUN: %clangxx_tsan %s -o %t_O0 -O0 -fobjc-arc -framework Foundation
// RUN: %deflake %run %t_O0 2>&1 | FileCheck %s

// RUN: %clangxx_tsan %s -o %t_O2 -O2 -fobjc-arc -framework Foundation
// RUN: %deflake %run %t_O2 2>&1 | FileCheck %s

#import "../test.h"
#import <Foundation/Foundation.h>

@interface CopyContainer : NSObject
@property(nonatomic, copy) NSString *text;
@property(nonatomic, copy) NSString *readWriteText;
@property(atomic, copy) NSString *atomicText;
@end

@implementation CopyContainer
@end

int main() {
  @autoreleasepool {
    CopyContainer *c = [CopyContainer new];
    NSString *s1 = [NSString stringWithFormat:@"hello-non-tagged-string-%d", 1];
    NSString *s2 = [NSString stringWithFormat:@"world-non-tagged-string-%d", 2];
    dispatch_queue_t q =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    // 1. Concurrent access to an (atomic, copy) property must not race.
    {
      barrier_init(&barrier, 2);
      dispatch_group_t group = dispatch_group_create();
      dispatch_group_async(group, q, ^{
        c.atomicText = s1;
        barrier_wait(&barrier);
      });
      dispatch_group_async(group, q, ^{
        barrier_wait(&barrier);
        c.atomicText = s2;
      });
      dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
      NSLog(@"Atomic write completed.");
    }
    // CHECK-NOT: WARNING: ThreadSanitizer
    // CHECK: Atomic write completed.

    // 2. Write-vs-Write race on a (nonatomic, copy) property.
    {
      barrier_init(&barrier, 2);
      dispatch_group_t group = dispatch_group_create();
      dispatch_group_async(group, q, ^{
        c.text = s1;
        barrier_wait(&barrier);
      });
      dispatch_group_async(group, q, ^{
        barrier_wait(&barrier);
        c.text = s2;
      });
      dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[CopyContainer setText:]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[CopyContainer setText:]

    // 3. Read-vs-Write race (getter vs setter) on a (nonatomic, copy) property.
    {
      c.readWriteText = s1;
      barrier_init(&barrier, 2);
      dispatch_group_t group = dispatch_group_create();
      dispatch_group_async(group, q, ^{
        c.readWriteText = s2;
        barrier_wait(&barrier);
      });
      dispatch_group_async(group, q, ^{
        barrier_wait(&barrier);
        (void)c.readWriteText;
      });
      dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }
    // CHECK: WARNING: ThreadSanitizer: data race
    // CHECK:   Read of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 -[CopyContainer readWriteText]
    // CHECK:   Previous write of size 8 at {{0x[0-9a-f]+}} by thread T{{[0-9]+}}{{.*}}:
    // CHECK:     #0 objc_setProperty_nonatomic_copy
    // CHECK-NEXT:     #1 -[CopyContainer setReadWriteText:]
  }
  return 0;
}
