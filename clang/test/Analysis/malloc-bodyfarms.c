// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker core,unix -verify %s

typedef __typeof(sizeof(int)) size_t;
void *calloc(size_t, size_t);

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_sync(dispatch_queue_t, dispatch_block_t);

void test_no_state_change_in_body_farm(dispatch_queue_t queue) {
  dispatch_sync(queue, ^{}); // no-crash
  calloc(1, 1);
} // expected-warning{{Potential memory leak}}

void test_no_state_change_in_body_farm_2(dispatch_queue_t queue) {
  void *p = calloc(1, 1);
  dispatch_sync(queue, ^{}); // no-crash
  p = 0;
} // expected-warning{{Potential leak of memory pointed to by 'p'}}
