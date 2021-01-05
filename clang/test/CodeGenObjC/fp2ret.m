// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-X86_32 %s
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-X86_64 %s
//
// RUN: %clang_cc1 -triple armv7-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -target-abi apcs-gnu -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-ARMV7 %s

@interface A
-(_Complex long double) complexLongDoubleValue;
@end


// CHECK-X86_32-LABEL: define{{.*}} void @t0()
// CHECK-X86_32: call void bitcast {{.*}} @objc_msgSend_stret to
// CHECK-X86_32: }
//
// CHECK-X86_64-LABEL: define{{.*}} void @t0()
// CHECK-X86_64: call { x86_fp80, x86_fp80 } bitcast {{.*}} @objc_msgSend_fp2ret to
// CHECK-X86_64: }
//
// CHECK-ARMV7-LABEL: define{{.*}} void @t0()
// CHECK-ARMV7: call i128 bitcast {{.*}} @objc_msgSend to
// CHECK-ARMV7: }
void t0() {
  [(A*)0 complexLongDoubleValue];
}
