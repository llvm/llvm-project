// RUN: %clang -fobjc-arc -Wno-objc-root-class -ObjC -fobjc-runtime=ios -FFoundation \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=1 \
// RUN: -target x86_64-apple-macosx10.15.0 -c -o - %s | \
// RUN: llvm-nm - | FileCheck -check-prefix=CHECK-WRAPPER %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s | llvm-nm - | \
// RUN: FileCheck -check-prefix=CHECK-DEFAULT %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=1 \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DEF %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class -DNO_OBJC_IMPL \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=1 \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DECLARE %s

// RUN: not %clang -fobjc-arc -Wno-objc-root-class -DENABLE_PROTOCOL_DIRECT_FAIL \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=1 \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s -S -emit-llvm 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-PROTOCOL-DIRECT-FAIL %s

////////////////////////////////////////////////////////////////////////////////

// RUN: %clang -fobjc-arc -Wno-objc-root-class -ObjC -fobjc-runtime=ios -FFoundation \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=0 \
// RUN: -target x86_64-apple-macosx10.15.0 -c -o - %s | \
// RUN: llvm-nm - | FileCheck -check-prefix=CHECK-WRAPPER-INDIRECT %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=0 \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DEF-INDIRECT %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class -DNO_OBJC_IMPL \
// RUN: -DENABLE_VISIBLE_OBJC_DIRECT=0 \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DECLARE-INDIRECT %s

// CHECK-WRAPPER: T _-<C testMethod:bar:>
         // TODO: Fix this
// CHECK-DEFAULT: t -[C testMethod:bar:]
// CHECK-WRAPPER-IR-DEF: define {{(dso_local )?}}void @"-<C testMethod:bar:>"
// CHECK-WRAPPER-IR-DECLARE: declare {{(dso_local )?}}void @"-<C testMethod:bar:>"
// CHECK-PROTOCOL-DIRECT-FAIL: error: 'objc_direct' attribute cannot be applied to methods declared in an Objective-C protocol

// CHECK-WRAPPER-INDIRECT-NOT: T _-<C testMethod:bar:>
// CHECK-WRAPPER-IR-DEF-INDIRECT-NOT: define {{(dso_local )?}}void @"-<C testMethod:bar:>"
// CHECK-WRAPPER-IR-DECLARE-INDIRECT-NOT: declare {{(dso_local )?}}void @"-<C testMethod:bar:>"

#if ENABLE_VISIBLE_OBJC_DIRECT
#define OBJC_DIRECT __attribute__((objc_direct)) __attribute__((visibility("default")))
#else
#define OBJC_DIRECT
#endif

@interface C
- (void)testMethod:(int)arg1 bar:(float)arg2 OBJC_DIRECT;
@end

#ifndef NO_OBJC_IMPL
@implementation C
- (void)testMethod:(int)arg1 bar:(float)arg2 OBJC_DIRECT {
}
@end
#endif

#ifdef ENABLE_PROTOCOL_DIRECT_FAIL
@protocol ProtoDirectVisibleFail
- (void)protoMethod OBJC_DIRECT;      // expected-error {{'objc_direct' attribute cannot be applied to methods declared in an Objective-C protocol}}
@end
#endif

C *c;

void f() {
  [c testMethod:1 bar:1.0];
}
