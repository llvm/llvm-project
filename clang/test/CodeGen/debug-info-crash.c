// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -debug-info-kind=limited -S %s -o -

// rdar://7590323
typedef struct dispatch_queue_s *dispatch_queue_t;
__attribute__((visibility("default")))
extern struct dispatch_queue_s _dispatch_main_q;
typedef struct dispatch_item_s *dispatch_item_t;
typedef void (^dispatch_legacy_block_t)(dispatch_item_t);
dispatch_item_t LEGACY_dispatch_call(dispatch_queue_t dq,
                                     dispatch_legacy_block_t dispatch_block,
                                     dispatch_legacy_block_t callback_block) {
  dispatch_queue_t lq = _dispatch_queue_get_current() ?: (&_dispatch_main_q);
  dispatch_async(dq, ^{
      if (callback_block) {
        dispatch_async(lq, ^{
          }
          );
      }
    }
    );
}

// radar://9008853
typedef struct P {
  int x;
} PS;
# 1 ""
void foo(void) {
  PS p2;
}
