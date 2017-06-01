// RUN: %clang -target armv7-windows -fobjc-runtime=ios -O1 -fexceptions -S -emit-llvm %s -o - | FileCheck %s
// REQUIRES: arm-registered-target

void (*f)(id);
void (*g)(void);
void h(void);

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)i;
@end

void i(void) {
  @try {
    @throw(@1);
  } @catch (id i) {
    (*f)(i);
    (*g)();
  }
}

// CHECK: call arm_aapcs_vfpcc i8* @objc_begin_catch
// CHECK: call arm_aapcs_vfpcc void @objc_end_catch
// CHECK-NOT: call i8* @objc_begin_catch
// CHECK-NOT: call void @objc_end_catch

