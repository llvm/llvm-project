// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-X86-64 %s

// radar 7547942
// Allow injection of ivars into implementation's implicit class.

@implementation INTFSTANDALONE // expected-warning {{cannot find interface declaration for 'INTFSTANDALONE'}}
{
  id IVAR1;
  id IVAR2;
}
- (id) Meth { return IVAR1; }
@end

// CHECK-X86-64: @"OBJC_IVAR_$_INTFSTANDALONE.IVAR1"
// CHECK-X86-64: @"OBJC_IVAR_$_INTFSTANDALONE.IVAR2"

