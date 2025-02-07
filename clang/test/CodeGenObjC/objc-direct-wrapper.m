// RUN: %clang -fobjc-arc -Wno-objc-root-class -ObjC -fobjc-runtime=ios -FFoundation \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN:   -fobjc-export-direct-methods -c -o - %s | \
// RUN: llvm-nm - | FileCheck -check-prefix=CHECK-WRAPPER %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation -c -o - %s | llvm-nm - | \
// RUN: FileCheck -check-prefix=CHECK-DEFAULT %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation \
// RUN:   -fobjc-export-direct-methods -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DEF %s

// RUN: %clang -fobjc-arc -Wno-objc-root-class -DNO_OBJC_IMPL \
// RUN: -target x86_64-apple-macosx10.15.0 \
// RUN: -ObjC -fobjc-runtime=ios -FFoundation \
// RUN:   -fobjc-export-direct-methods -c -o - %s -S -emit-llvm | \
// RUN: FileCheck -check-prefix=CHECK-WRAPPER-IR-DECLARE %s

// CHECK-WRAPPER: t -[C testNonDirectMethod:bar:]
// CHECK-WRAPPER: T _-<C testMethod:bar:>
// TODO: Fix this
// CHECK-DEFAULT: T -[C testMethod:bar:]
// CHECK-DEFAULT: t -[C testNonDirectMethod:bar:]
// CHECK-WRAPPER-IR-DEF: define {{(dso_local )?}}void @"-<C testMethod:bar:>"
// CHECK-WRAPPER-IR-DEF: define {{(internal )?}}void @"\01-[C testNonDirectMethod:bar:]"
// CHECK-WRAPPER-IR-DECLARE: declare {{(dso_local )?}}void @"-<C testMethod:bar:>"
// CHECK-WRAPPER-IR-DECLARE-NOT: declare {{(internal )?}}void @"\01-[C testNonDirectMethod:bar:]"
// CHECK-WRAPPER-IR-DECLARE: declare ptr @objc_msgSend

@interface C
- (void)testMethod:(int)arg1 bar:(float)arg2 __attribute((objc_direct));
- (void)testNonDirectMethod:(int)arg1 bar:(float)arg2;
@end

#ifndef NO_OBJC_IMPL
@implementation C
- (void)testMethod:(int)arg1 bar:(float)arg2 __attribute((objc_direct)) {
}
- (void)testNonDirectMethod:(int)arg1 bar:(float)arg2 {
}
@end
#endif

C *c;

void f() {
  [c testMethod:1 bar:1.0];
  [c testNonDirectMethod:1 bar:1.0];
}
