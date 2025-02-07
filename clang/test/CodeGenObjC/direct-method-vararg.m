// RUN: %clang -fobjc-export-direct-methods     \
// RUN:   -target arm64-apple-darwin -fobjc-arc \
// RUN:   -O0 -S -emit-llvm %s -o - | FileCheck %s


#include <stdio.h>
#include <stdarg.h>
__attribute__((objc_root_class))
@interface Root
- (void)printWithFormat:(const char *)format, ... __attribute__((objc_direct));
- (void)vprintWithFormat:(const char *)format Args:(va_list) args __attribute__((objc_direct));
@end

@implementation Root
// CHECK-LABEL: define {{.*}} void @"-<Root printWithFormat:>"
- (void)printWithFormat:(const char *)format, ... {
  // Inner functions won't be called since var arg functions don't have a thunk.
  // CHECK: call void (ptr, ptr, ...) @"-<Root printWithFormat:>"
  // CHECK: call void (ptr, ptr, ...) @"-<Root printWithFormat:>"
  [self printWithFormat:format, "Hello World"];
  [self printWithFormat:format, "!", 1, 2.0];
  va_list args;
  // CHECK: call void @llvm.va_start
  va_start(args, format);
  // CHECK: call void @"-<Root vprintWithFormat:Args:>_inner"
  [self vprintWithFormat:format Args:args];
  // CHECK: call void @llvm.va_end
  va_end(args);
}
// CHECK-NOT: <Root printWithFormat>_inner

// CHECK-LABEL: define {{.*}} void @"-<Root vprintWithFormat:Args:>"
// CHECK: call void @"-<Root vprintWithFormat:Args:>_inner"

// CHECK-LABEL: define {{.*}} void @"-<Root vprintWithFormat:Args:>_inner"
-(void)vprintWithFormat:(const char *)format Args:(va_list) args{
  // CHECK: call void @"-<Root vprintWithFormat:Args:>_inner"
  [self vprintWithFormat:format Args:args];
  // CHECK: call i32 @vprintf
  vprintf(format, args);
}
@end

void printRoot(Root* root, const char* format, ...) {
  // CHECK: call void (ptr, ptr, ...) @"-<Root printWithFormat:>"(ptr {{.*}}, ptr {{.*}})
  [root printWithFormat:"Hello World"];
  // CHECK: call void (ptr, ptr, ...) @"-<Root printWithFormat:>"(ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i32 noundef 1, double noundef 2.000000e+00)
  [root printWithFormat:"Hello World%s %d %lf", "!", 1, 2.0];

  // CHECK: call void @"-<Root vprintWithFormat:Args:>"(ptr {{.*}}, ptr {{.*}}, ptr {{.*}} null)
  [root vprintWithFormat:"Hello World" Args:NULL];
  va_list args;
  // CHECK: call void @llvm.va_start
  va_start(args, format);
  // CHECK: call void @"-<Root vprintWithFormat:Args:>"(ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
  [root vprintWithFormat:format Args:args];
  // CHECK: call void @llvm.va_end
  va_end(args);
}
