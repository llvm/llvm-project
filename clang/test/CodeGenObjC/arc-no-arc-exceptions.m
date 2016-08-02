// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fblocks -fexceptions -fobjc-exceptions -O2 -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fblocks -fexceptions -fobjc-exceptions -disable-llvm-optzns -o - %s | FileCheck -check-prefix=NO-METADATA %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fblocks -fexceptions -fobjc-exceptions -O2 -disable-llvm-optzns -o - %s -fobjc-arc-exceptions | FileCheck -check-prefix=NO-METADATA %s

// The front-end should emit clang.arc.no_objc_arc_exceptions in -fobjc-arc-exceptions
// mode when optimization is enabled, and not otherwise.

void thrower(void);
void not(void) __attribute__((nothrow));

// CHECK-LABEL: define void @test0(
// CHECK: call void @thrower(), !clang.arc.no_objc_arc_exceptions !
// CHECK: call void @not() [[NUW:#[0-9]+]], !clang.arc.no_objc_arc_exceptions !
// NO-METADATA-LABEL: define void @test0(
// NO-METADATA-NOT: !clang.arc.no_objc_arc_exceptions
// NO-METADATA: }
void test0(void) {
  thrower();
  not();
}

// CHECK-LABEL: define void @test1(
// CHECK: call void @thrower(), !clang.arc.no_objc_arc_exceptions !
// CHECK: call void @not() [[NUW]], !clang.arc.no_objc_arc_exceptions !
// NO-METADATA-LABEL: define void @test1(
// NO-METADATA-NOT: !clang.arc.no_objc_arc_exceptions
// NO-METADATA: }
void test1(id x) {
  id y = x;
  thrower();
  not();
}

void NSLog(id, ...);

// CHECK-LABEL: define void @test2(
// CHECK: invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*), i32* %{{.*}})
// CHECK:   to label %{{.*}} unwind label %{{.*}}, !clang.arc.no_objc_arc_exceptions !
// NO-METADATA-LABEL: define void @test2(
// NO-METADATA-NOT: !clang.arc.no_objc_arc_exceptions
// NO-METADATA: }
void test2(void) {
    @autoreleasepool {
        __attribute__((__blocks__(byref))) int x;
        NSLog(@"Address of x outside of block: %p", &x);
    }
}

// CHECK-LABEL: define void @test3(
// CHECK: invoke void %{{.*}}(i8* %{{.*}})
// CHECK:   to label %{{.*}} unwind label %{{.*}}, !clang.arc.no_objc_arc_exceptions !
// NO-METADATA-LABEL: define void @test3(
// NO-METADATA-NOT: !clang.arc.no_objc_arc_exceptions
// NO-METADATA: }
void test3(void) {
    @autoreleasepool {
        __attribute__((__blocks__(byref))) int x;
        ^{
            NSLog(@"Address of x in non-assigned block: %p", &x);
        }();
    }
}

// CHECK-LABEL: define void @test4(
// CHECK: invoke void %{{.*}}(i8* %{{.*}})
// CHECK:   to label %{{.*}} unwind label %{{.*}}, !clang.arc.no_objc_arc_exceptions !
// NO-METADATA-LABEL: define void @test4(
// NO-METADATA-NOT: !clang.arc.no_objc_arc_exceptions
// NO-METADATA: }
void test4(void) {
    @autoreleasepool {
        __attribute__((__blocks__(byref))) int x;
        void (^b)(void) = ^{
            NSLog(@"Address of x in assigned block: %p", &x);
        };
        b();
    }
}

// CHECK: attributes [[NUW]] = { nounwind }
