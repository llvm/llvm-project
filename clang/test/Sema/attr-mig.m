// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

typedef int kern_return_t;
#define KERN_SUCCESS 0

@interface NSObject
@end

@interface I: NSObject
- (kern_return_t)foo __attribute__((mig_server_routine)); // no-warning
- (void) bar_void __attribute__((mig_server_routine)); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
- (int) bar_int __attribute__((mig_server_routine)); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
@end

@implementation I
- (kern_return_t)foo {
  kern_return_t (^block)(void) = ^ __attribute__((mig_server_routine)) { // no-warning
    return KERN_SUCCESS;
  };

  // FIXME: Warn that this block doesn't return a kern_return_t.
  void (^invalid_block)(void) = ^ __attribute__((mig_server_routine)) {};

  return block();
}
- (void)bar_void {
}
- (int)bar_int {
  return 0;
}
@end
