// RUN: %clang_cc1 -triple x86_64-apple-macos11 -fsyntax-only -fobjc-arc -fblocks -verify -Wunused-but-set-variable -Wno-objc-root-class %s

typedef struct dispatch_queue_s *dispatch_queue_t;

typedef void (^dispatch_block_t)(void);

void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);

extern __attribute__((visibility("default"))) struct dispatch_queue_s _dispatch_main_q;

id getFoo(void);

@protocol P

@end

@interface I

@end

void test(void) {
  // no diagnostics
  __block id x = getFoo();
  __block id<P> y = x;
  __block I *z = (I *)x;
  // diagnose non-block variables
  id x2 = getFoo(); // expected-warning {{variable 'x2' set but not used}}
  dispatch_async(&_dispatch_main_q, ^{
    x = ((void *)0);
    y = x;
    z = ((void *)0);
  });
  x2 = getFoo();
}
