// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -fblocks -fobjc-arc -verify %s

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);
typedef long dispatch_time_t;
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue, dispatch_block_t block);

extern dispatch_queue_t queue;
extern dispatch_once_t *predicate;
extern dispatch_time_t when;

void test_block_expr_async() {
  int x = 123;
  int *p = &x;

  dispatch_async(queue, ^{
    *p = 321;
  });
  // expected-warning@-3 {{Address of stack memory associated with local variable 'x' \
is captured by an asynchronously-executed block}}
}

void test_block_expr_once_no_leak() {
  int x = 123;
  int *p = &x;
  // synchronous, no warning
  dispatch_once(predicate, ^{
    *p = 321;
  });
}

void test_block_expr_after() {
  int x = 123;
  int *p = &x;
  dispatch_after(when, queue, ^{
    *p = 321;
  });
  // expected-warning@-3 {{Address of stack memory associated with local variable 'x' \
is captured by an asynchronously-executed block}}
}

void test_block_expr_async_no_leak() {
  int x = 123;
  int *p = &x;
  // no leak
  dispatch_async(queue, ^{
    int y = x;
    ++y;
  });
}

void test_block_var_async() {
  int x = 123;
  int *p = &x;
  void (^b)(void) = ^void(void) {
    *p = 1; 
  };
  dispatch_async(queue, b);
  // expected-warning@-1 {{Address of stack memory associated with local variable 'x' \
is captured by an asynchronously-executed block}}
}

void test_block_with_ref_async() {
  int x = 123;
  int &r = x;
  void (^b)(void) = ^void(void) {
    r = 1; 
  };
  dispatch_async(queue, b);
  // expected-warning@-1 {{Address of stack memory associated with local variable 'x' \
is captured by an asynchronously-executed block}}
}

dispatch_block_t get_leaking_block() {
  int leaked_x = 791;
  int *p = &leaked_x;
  return ^void(void) {
    *p = 1; 
  };
  // expected-warning@-3 {{Address of stack memory associated with local variable 'leaked_x' \
is captured by a returned block}}
}

void test_returned_from_func_block_async() {
  dispatch_async(queue, get_leaking_block());
  // expected-warning@-1 {{Address of stack memory associated with local variable 'leaked_x' \
is captured by an asynchronously-executed block}}
}

// synchronous, no leak
void test_block_var_once() {
  int x = 123;
  int *p = &x;
  void (^b)(void) = ^void(void) {
    *p = 1; 
  };
  dispatch_once(predicate, b); // no-warning
}

void test_block_var_after() {
  int x = 123;
  int *p = &x;
  void (^b)(void) = ^void(void) {
    *p = 1; 
  };
  dispatch_after(when, queue, b);
  // expected-warning@-1 {{Address of stack memory associated with local variable 'x' \
is captured by an asynchronously-executed block}}
}

void test_block_var_async_no_leak() {
  int x = 123;
  int *p = &x;
  void (^b)(void) = ^void(void) {
    int y = x;
    ++y; 
  };
  dispatch_async(queue, b); // no-warning
}

void test_block_inside_block_async_no_leak() {
  int x = 123;
  int *p = &x;
  void (^inner)(void) = ^void(void) {
    int y = x;
    ++y; 
  };
  void (^outer)(void) = ^void(void) {
    int z = x;
    ++z;
    inner(); 
  };
  dispatch_async(queue, outer); // no-warning
}

dispatch_block_t accept_and_pass_back_block(dispatch_block_t block) {
  block();
  return block; // no-warning
}

void test_passing_continuation_no_leak() {
  int x = 123;
  int *p = &x;
  void (^cont)(void) = ^void(void) {
    *p = 128;
  };
  accept_and_pass_back_block(cont); // no-warning
}

@interface NSObject
@end
@protocol OS_dispatch_semaphore
@end
typedef NSObject<OS_dispatch_semaphore> *dispatch_semaphore_t;
dispatch_semaphore_t dispatch_semaphore_create(long value);
long dispatch_semaphore_wait(dispatch_semaphore_t dsema, dispatch_time_t timeout);
long dispatch_semaphore_signal(dispatch_semaphore_t dsema);

void test_no_leaks_on_semaphore_pattern() {
  int x = 0;
  int *p = &x;
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  dispatch_async(queue, ^{
    *p = 1;
    // Some work.
    dispatch_semaphore_signal(semaphore);
  }); // no-warning

  // Do some other work concurrently with the asynchronous work
  // Wait for the asynchronous work to finish
  dispatch_semaphore_wait(semaphore, 1000);
}
