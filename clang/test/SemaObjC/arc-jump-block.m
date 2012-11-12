// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify -Wno-objc-root-class %s
// rdar://9535237

typedef struct dispatch_queue_s *dispatch_queue_t;

typedef void (^dispatch_block_t)(void);

void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);

extern __attribute__((visibility("default"))) struct dispatch_queue_s _dispatch_main_q;

@interface SwitchBlockCrashAppDelegate
- (void)pageLeft;
- (void)pageRight;;
@end

@implementation SwitchBlockCrashAppDelegate

- (void)choose:(int)button {
    switch (button) {
    case 0:
        dispatch_async((&_dispatch_main_q), ^{ [self pageLeft]; }); // expected-note 3 {{jump enters lifetime of block which strongly captures a variable}}
        break;
    case 2:  // expected-error {{switch case is in protected scope}}
        dispatch_async((&_dispatch_main_q), ^{ [self pageRight]; }); // expected-note 2 {{jump enters lifetime of block which strongly captures a variable}}
        break;
    case 3: // expected-error {{switch case is in protected scope}}
        {
          dispatch_async((&_dispatch_main_q), ^{ [self pageRight]; });
          break;
        }
    case 4: // expected-error {{switch case is in protected scope}}
        break;
    }

    __block SwitchBlockCrashAppDelegate *captured_block_obj;
    switch (button) {
    case 10:
      {
        dispatch_async((&_dispatch_main_q), ^{ [self pageLeft]; });
        break;
      }
    case 12:
        if (button)
          dispatch_async((&_dispatch_main_q), ^{ [captured_block_obj pageRight]; });
        break;
    case 13:
        while (button)
          dispatch_async((&_dispatch_main_q), ^{ [self pageRight]; });
        break;
    case 14:
        break;
    }

    switch (button) {
    case 10:
      {
        dispatch_async((&_dispatch_main_q), ^{ [self pageLeft]; });
        break;
      }
    case 12:
        if (button)
          dispatch_async((&_dispatch_main_q), ^{ [self pageRight]; });
        switch (button) {
          case 0:
            {
              dispatch_async((&_dispatch_main_q), ^{ [self pageLeft]; });
              break;
            }
         case 4: 
          break;
        }
        break;
    case 13:
        while (button)
          dispatch_async((&_dispatch_main_q), ^{ [self pageRight]; });
        break;
    case 14:
        break;
    }
}
- (void)pageLeft {}
- (void)pageRight {}
@end

// Test 2.  rdar://problem/11150919
int test2(id obj, int state) { // expected-note {{jump enters lifetime of block}} FIXME: wierd location
  switch (state) {
  case 0:
    (void) ^{ (void) obj; };
    return 0;

  default: // expected-error {{switch case is in protected scope}}
    return 1;
  }
}

