// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// CHECK: @"OBJC_IVAR_$_I3._iv2" = global i64 8, section "__DATA, __objc_ivar", align 8
// CHECK: @"OBJC_IVAR_$_I3._iv3" = global i64 12, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I3" = private global {{.*}} { i32 0, i32 8, i32 13
// CHECK: @"OBJC_IVAR_$_I4._iv4" = global i64 13, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I4" = private global {{.*}} { i32 0, i32 13, i32 14, {{.*}}
// CHECK: @"OBJC_IVAR_$_I5._iv6_synth" = hidden global i64 16, section "__DATA, __objc_ivar", align 8
// CHECK: @"OBJC_IVAR_$_I5._iv7_synth" = hidden global i64 20, section "__DATA, __objc_ivar", align 8
// CHECK: @"OBJC_IVAR_$_I5._iv5" = global i64 14, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I5" = private global {{.*}} { i32 0, i32 14, i32 24, {{.*}}
// CHECK: @"OBJC_IVAR_$_I6.iv0" = global i64 0, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I6" = private global {{.*}} { i32 2, i32 0, i32 1, {{.*}}
// CHECK: @"OBJC_IVAR_$_I8.b" = global i64 8, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I8" = private global {{.*}} { i32 0, i32 8, i32 16, {{.*}}
// CHECK: @"OBJC_IVAR_$_I9.iv0" = global i64 0, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I9" = private global {{.*}} { i32 2, i32 0, i32 4, {{.*}}
// CHECK: @"OBJC_IVAR_$_I10.iv1" = global i64 4, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I10" = private global {{.*}} { i32 0, i32 4, i32 5, {{.*}}
// CHECK: _OBJC_CLASS_RO_$_I11" = private global {{.*}} { i32 0, i32 5, i32 5, {{.*}}
// CHECK: @"OBJC_IVAR_$_I12.iv2" = global i64 8, section "__DATA, __objc_ivar", align 8
// CHECK: _OBJC_CLASS_RO_$_I12" = private global {{.*}} { i32 0, i32 8, i32 12, {{.*}}

/*
  Compare to:
    gcc -m64 -S -o - interface-layout-64.m | grep '^_OBJC_IVAR_$_*{{.*}}' -A 1
  and 
    gcc -m64 -S -o - interface-layout-64.m | grep '^l{{.*}}_CLASS_RO_$_I[0-9]*' -A 3
 */

struct s0 {
  double x;
};

@interface I2 {
  struct s0 _iv1;
}
@end

@interface I3 : I2 {
  unsigned int _iv2 :1;
  unsigned : 0;
  unsigned int _iv3 : 3;
}
@end

@interface I4 : I3 {
 char _iv4;
}
@end

@interface I5 : I4 {
 char _iv5;
}

@property int prop0;
@end

@implementation I3
@end

@implementation I4 
@end

@interface I5 ()
@property int prop1;
@property char prop2;
@end

@implementation I5
@synthesize prop0 = _iv6_synth;
@synthesize prop1 = _iv7_synth;
@synthesize prop2 = _iv5;
@end

// The size rounds up to the next available byte.
@interface I6 {
  unsigned iv0 : 2;
}
@end
@implementation I6
@end

// The start of the subclass includes padding for its own alignment.
@interface I7 {
  char a;
}
@end
@interface I8 : I7 {
  double b;
}
@end
@implementation I8
@end

// Padding bit-fields
@interface I9 {
  unsigned iv0 : 2;
  unsigned : 0;
}
@end
@implementation I9
@end
@interface I10 : I9 {
  unsigned iv1 : 2;
}
@end
@implementation I10
@end

// Empty structures
@interface I11 : I10
@end
@implementation I11
@end
@interface I12 : I11 {
  unsigned iv2;
}
@end
@implementation I12
@end
