// RUN: %clang_cc1 -x objective-c -triple x86_64-apple-darwin10 %S/Common.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -include-pch %t.pch -arcmt-action=check -verify -triple x86_64-apple-darwin10 -fblocks -Werror %s
// REQUIRES: x86-registered-target

@interface I9601437 {
  __unsafe_unretained id x;
}
-(void)Meth;
@end

@implementation I9601437
-(void)Meth {
  self->x = [NSObject new]; // expected-error {{assigning retained object}}
}
@end
