// RUN: %clang_tsan %s -lc++ -fobjc-arc -lobjc -o %t -framework Foundation %darwin_min_target_with_full_runtime_arc_support
// RUN: %run %t 2>&1 | FileCheck %s

// Check that we do not report races between:
// - Object retain and initialize
// - Object release and dealloc
// - Object release and .cxx_destruct

#import <Foundation/Foundation.h>
#include "../test.h"
invisible_barrier_t barrier2;

class NeedCleanup {
  public:
    int x;
    NeedCleanup() {
      x = 1;
    }
    ~NeedCleanup() {
      x = 0;
    }
};

@interface TestDeallocObject : NSObject {
  @public
    int v;
  }
  - (id)init;
  - (void)accessMember;
  - (void)dealloc;
@end

@implementation TestDeallocObject
  - (id)init {
    if ([super self]) {
      v = 1;
      return self;
    }
    return nil;
  }
  - (void)accessMember {
    int local = v;
    local++;
  }
  - (void)dealloc {
    v = 0;
  }
@end

@interface TestCXXDestructObject : NSObject {
  @public
    NeedCleanup cxxMemberWithCleanup;
  }
  - (void)accessMember;
@end

@implementation TestCXXDestructObject
  - (void)accessMember {
    int local = cxxMemberWithCleanup.x;
    local++;
  }
@end

@interface TestInitializeObject : NSObject
@end

@implementation TestInitializeObject
  static long InitializerAccessedGlobal = 0;
  + (void)initialize {
      InitializerAccessedGlobal = 42;
  }
@end

int main(int argc, const char *argv[]) {
  // Ensure that there is no race when calling initialize on TestInitializeObject;
  // otherwise, the locking from ObjC runtime becomes observable. Also ensures that
  // blocks are dispatched to 2 different threads.
  barrier_init(&barrier, 2);
  // Ensure that objects are destructed during block object release.
  barrier_init(&barrier2, 3);

  TestDeallocObject *tdo = [[TestDeallocObject alloc] init];
  TestCXXDestructObject *tcxxdo = [[TestCXXDestructObject alloc] init];
  [tdo accessMember];
  [tcxxdo accessMember];
  {
    dispatch_queue_t q = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
    dispatch_async(q, ^{
        [TestInitializeObject new];
        barrier_wait(&barrier);
        long local = InitializerAccessedGlobal;
        local++;
        [tdo accessMember];
        [tcxxdo accessMember];
        barrier_wait(&barrier2);
    });
    dispatch_async(q, ^{
        barrier_wait(&barrier);
        [TestInitializeObject new];
        long local = InitializerAccessedGlobal;
        local++;
        [tdo accessMember];
        [tcxxdo accessMember];
        barrier_wait(&barrier2);
    });
  }
  barrier_wait(&barrier2);
  NSLog(@"Done.");
  return 0;
}

// CHECK: Done.
// CHECK-NOT: ThreadSanitizer: data race
