// RUN: %clang_cc1 %s -emit-llvm -fsanitize=address -o - | FileCheck %s

@interface I0 @end
@implementation I0
// CHECK-NOT: sanitize_address
- (void) im0: (int) a0 __attribute__((no_sanitize("address"))) {
}
@end
