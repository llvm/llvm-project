// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,alpha.core.StackAddressAsyncEscape -fblocks -verify %s

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
extern dispatch_queue_t queue;
void f(int);

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

// The block literal defined in this function could leak once being
// called.
void output_block(dispatch_block_t * blk) {
  int x = 0;
  *blk = ^{ f(x); }; // expected-warning {{Address of stack-allocated block declared on line 43 is still referred to by the stack variable 'blk' upon returning to the caller.  This will be a dangling reference [core.StackAddressEscape]}}
}

// The block literal captures nothing thus is treated as a constant.
void output_constant_block(dispatch_block_t * blk) {
  *blk = ^{ };
}

// A block can leak if it captures at least one variable and is not
// under ARC when its' stack frame expires.
void test_block_leak() {
  __block dispatch_block_t blk;
  int x = 0;
  dispatch_block_t p = ^{
    blk = ^{ // expected-warning {{Address of stack-allocated block declared on line 57 is still referred to by the stack variable 'blk' upon returning to the caller.  This will be a dangling reference [core.StackAddressEscape]}}
      f(x);
    };
  };

  p();
  blk();
  output_block(&blk);
  blk();
}

// A block captures nothing is a constant thus never leaks.
void test_constant_block_no_leak() {
  __block dispatch_block_t blk;
  dispatch_block_t p = ^{
    blk = ^{
      f(0);
    };
  };
  
  p();
  blk();
  output_constant_block(&blk);
  blk();
}
