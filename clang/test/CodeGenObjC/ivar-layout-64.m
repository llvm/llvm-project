// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o - %s | FileCheck %s

/*

Here is a handy command for looking at llvm-gcc's output:
llvm-gcc -m64 -fobjc-gc -emit-llvm -S -o - ivar-layout-64.m | \
  grep 'OBJC_CLASS_NAME.* =.*global' | \
  sed -e 's#, section.*# ...#' | \
  sed -e 's#_[0-9]*"#_NNN#' | \
  sort

*/

@interface B @end

@interface A {
  struct s0 {
    int f0;
    int f1;
  } f0;
  id f1;
__weak B *f2;
  int f3 : 5;
  struct s1 {
    int *f0;
    int *f1;
  } f4[2][1];
}
@end

@interface C : A
@property int p3;
@end

// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"C\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"\11p\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"!`\00"


@implementation C
@synthesize p3 = _p3;
@end

@interface A()
@property int p0;
@property (assign) __strong id p1;
@property (assign) __weak id p2;
@end

// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"A\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"\11q\10\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"!q\00"

@implementation A
@synthesize p0 = _p0;
@synthesize p1 = _p1;
@synthesize p2 = _p2;
@end

@interface D : A
@property int p3;
@end

// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"D\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"\11p\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"!`\00"

@implementation D
@synthesize p3 = _p3;
@end

typedef unsigned short UInt16;


typedef signed char BOOL;
typedef unsigned int FSCatalogInfoBitmap;

@interface NSFileLocationComponent {
    @private

    id _specifierOrStandardizedPath;
    BOOL _carbonCatalogInfoAndNameAreValid;
    FSCatalogInfoBitmap _carbonCatalogInfoMask;
    id _name;
    id _containerComponent;
    id _presentableName;
    id _iconAsAttributedString;
}
@end

// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"NSFileLocationComponent\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"\01\14\00"

@implementation NSFileLocationComponent @end

@interface NSObject {
  id isa;
}
@end

@interface Foo : NSObject {
    id ivar;

    unsigned long bitfield  :31;
    unsigned long bitfield2 :1;
    unsigned long bitfield3 :32;
}
@end

// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"Foo\00"
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global {{.*}} c"\02\10\00"

@implementation Foo @end
