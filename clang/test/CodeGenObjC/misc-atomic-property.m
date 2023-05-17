// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck %s
// rdar: //8808439

typedef struct {
#ifdef __LP64__
	unsigned char b[15];
#else
	unsigned char b[7];
#endif
} bools_minus_one;

typedef struct {
#ifdef __LP64__
	unsigned char b[16];
#else
	unsigned char b[8];
#endif
} bools;


@interface Foo
{
#ifndef __LP64__
       bools x;
       bools_minus_one y;
#endif
}
@property(assign) bools bools_p;
@property(assign) bools_minus_one bools_minus_one_p;
@end

@implementation Foo
@synthesize bools_p=x;
@synthesize bools_minus_one_p=y;
@end

#ifdef __LP64__
typedef __int128_t dword;
#else
typedef long long int dword;
#endif

@interface Test_dwords
{
#ifndef __LP64__
       dword dw;
#endif
}
@property(assign) dword dword_p;
@end

@implementation Test_dwords
@synthesize dword_p=dw;
@end


@interface Test_floats
{
  float fl;
  double d;
  long double ld;
}
@property(assign) float fl_p;
@property(assign) double  d_p;
@property(assign) long double ld_p;
@end

@implementation Test_floats
@synthesize fl_p = fl;
@synthesize d_p = d;
@synthesize ld_p = ld;
@end

// CHECK: call void @objc_copyStruct
// CHECK: call void @objc_copyStruct
// CHECK: call void @objc_copyStruct
// CHECK: call void @objc_copyStruct
// CHECK: call void @objc_copyStruct
// CHECK: call void @objc_copyStruct
