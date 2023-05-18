// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=gcc -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-1.7 -o - %s | FileCheck -check-prefix=NEW-ABI %s

void opaque(void);
void log(int i);

@class C;

// CHECK: define{{.*}} void @test0() [[TF:#[0-9]+]]
// CHECK-SAME: personality ptr @__gnu_objc_personality_v0
void test0(void) {
  @try {
    // CHECK: invoke void @opaque()
    opaque();

    // CHECK: call void @log(i32 noundef 1)

  } @catch (C *c) {
    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   catch ptr @0
    // CHECK:      br i1

    // CHECK: call void @log(i32 noundef 0)

    // CHECK: resume
    // NEW-ABI: objc_begin_catch
    // NEW-ABI: objc_end_catch

    log(0);
  }

  log(1);
}

// CHECK: attributes [[TF]] = { noinline optnone "{{.*}} }
