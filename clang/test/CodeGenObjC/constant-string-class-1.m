// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fno-constant-cfstrings -fconstant-string-class OFConstantString  -emit-llvm -o %t %s
// pr9914

@interface OFConstantString 
+ class;
@end

@interface OFString
- (void)XMLElementBySerializing;
@end

@implementation OFString

- (void)XMLElementBySerializing
{
 id str = @"object";

 [OFConstantString class];
}

@end

// CHECK: @"OBJC_CLASS_$_OFConstantString" = external global
