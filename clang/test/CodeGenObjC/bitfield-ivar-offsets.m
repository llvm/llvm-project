// RUNX: llvm-gcc -m64  -emit-llvm -S -o %t %s &&
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
// RUN: grep -F '@"OBJC_IVAR_$_I0._b0" = global i64 0, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._b1" = global i64 0, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._b2" = global i64 1, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._x" = global i64 2, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._b3" = global i64 4, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._y" = global i64 6, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0._b4" = global i64 7, section "__DATA, __objc_ivar", align 8' %t
// RUN: grep -F '@"OBJC_IVAR_$_I0." = global' %t | count 0

@interface I0 {
  unsigned _b0:4;
  unsigned _b1:5;
  unsigned _b2:5;
  char _x;
  unsigned _b3:9;
  char _y;
  char _b4:3;
  char : 0;
}
@end

@implementation I0
@end
