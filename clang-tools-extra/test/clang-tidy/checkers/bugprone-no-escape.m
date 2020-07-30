// RUN: %check_clang_tidy %s bugprone-no-escape %t
// RUN: %check_clang_tidy %s -assume-filename=bugprone-no-escape.c bugprone-no-escape %t -- -- -fblocks

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef struct dispatch_time_s *dispatch_time_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue, dispatch_block_t block);

extern dispatch_queue_t queue;

void test_noescape_attribute(__attribute__((noescape)) int *p, int *q) {
  dispatch_async(queue, ^{
    *p = 123;
    // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: pointer 'p' with attribute 'noescape' is captured by an asynchronously-executed block [bugprone-no-escape]
    // CHECK-MESSAGES: :[[@LINE-4]]:30: note: the 'noescape' attribute is declared here.
  });

  dispatch_after(456, queue, ^{
    *p = 789;
    // CHECK-MESSAGES: :[[@LINE-2]]:30: warning: pointer 'p' with attribute 'noescape' is captured by an asynchronously-executed block [bugprone-no-escape]
  });

  dispatch_async(queue, ^{
    *q = 0;
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:25: warning: pointer 'q' with attribute 'noescape' is captured by an asynchronously-executed block
  });
}
