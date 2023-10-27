// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin9 %s -emit-llvm -o - | FileCheck %s -check-prefix=X86
// RUN: %clang_cc1 -fblocks -triple arm-apple-darwin %s -emit-llvm -o - | FileCheck %s -check-prefix=ARM
// RUN: %clang_cc1 -fblocks -triple arm64-apple-darwin %s -emit-llvm -o - | FileCheck %s -check-prefix=ARM64

// Don't use 'stret' variants on ARM64.

// X86: @main
// X86: @objc_msgSend_stret

// ARM: @main
// ARM: @objc_msgSend_stret

// ARM64:     @main
// ARM64-NOT: @objc_msgSend_stret

struct st { int i[1000]; };
@interface Test
+(struct st)method;
@end
int main(void) {
  [Test method];
}
