// Check that dispatch_apply() does not report a false data race when the
// dispatched block's copy helper retains the captures into the heap block.
// NOTE: this test may spuriously pass, but is a best effort to reproduce
//       the problem, which relies on OS thread scheduling mechanics.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

int main() {
  // Strong capture -> non-trivial copy helper. Enough iterations that
  // libdispatch fans out to worker threads (captures read off-thread).
  const size_t n = 1024;
  NSMutableArray *items = [NSMutableArray array];
  for (size_t i = 0; i < n; i++)
    [items addObject:@(i)];

  dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(size_t idx) {
                   (void)[items objectAtIndex:idx];
                 });

  NSLog(@"Done.");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
