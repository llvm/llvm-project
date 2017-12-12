// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,alpha.core.StackAddressAsyncEscape -fblocks -verify %s

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
extern dispatch_queue_t queue;

void test_block_inside_block_async_no_leak() {
  int x = 123;
  int *p = &x;
  void (^inner)(void) = ^void(void) {
    int y = x;
    ++y; 
  };
  // Block_copy(...) copies the captured block ("inner") too,
  // there is no leak in this case.
  dispatch_async(queue, ^void(void) {
    int z = x;
    ++z;
    inner(); 
  }); // no-warning
}

dispatch_block_t test_block_inside_block_async_leak() {
  int x = 123;
  void (^inner)(void) = ^void(void) {
    int y = x;
    ++y; 
  };
  void (^outer)(void) = ^void(void) {
    int z = x;
    ++z;
    inner(); 
  }; 
  return outer; // expected-warning-re{{Address of stack-allocated block declared on line {{.+}} is captured by a returned block}}
}

