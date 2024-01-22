// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix CHECK-DWARF %s

// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck -check-prefix CHECK-MINGW %s

// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck -check-prefix CHECK-MSVC %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck -check-prefix CHECK-ELF %s

typedef struct {} Z;

@interface A
-(void)bar:(Z)a;
-(void)foo:(Z)a : (char*)b : (Z)c : (double) d;
@end

@implementation A
-(void)bar:(Z)a {}
-(void)foo:(Z)a: (char*)b : (Z)c : (double) d {}
@end

// CHECK-DWARF: private unnamed_addr constant [14 x i8] c"v16@0:8{?=}16
// CHECK-DWARF: private unnamed_addr constant [26 x i8] c"v32@0:8{?=}16*16{?=}24d24

// CHECK-MINGW: @".objc_sel_types_v16@0:8{?\02}16" = linkonce_odr hidden constant [14 x i8] c"v16@0:8{?=}16\00"
// CHECK-MINGW: @".objc_sel_types_v32@0:8{?\02}16*16{?\02}24d24" = linkonce_odr hidden constant [26 x i8] c"v32@0:8{?=}16*16{?=}24d24\00"

// CHECK-MSVC: @".objc_sel_types_v20@0:8{?\02}16" = linkonce_odr hidden constant [14 x i8] c"v20@0:8{?=}16\00"
// CHECK-MSVC: @".objc_sel_types_v40@0:8{?\02}16*20{?\02}28d32" = linkonce_odr hidden constant [26 x i8] c"v40@0:8{?=}16*20{?=}28d32\00"

// CHECK-ELF: @".objc_sel_types_v16\010:8{?=}16" = linkonce_odr hidden constant [14 x i8] c"v16@0:8{?=}16\00"
// CHECK-ELF: @".objc_sel_types_v32\010:8{?=}16*16{?=}24d24" = linkonce_odr hidden constant [26 x i8] c"v32@0:8{?=}16*16{?=}24d24\00"

@interface NSObject @end

@class BABugExample;
typedef BABugExample BABugExampleRedefinition;

@interface BABugExample : NSObject {
    BABugExampleRedefinition *_property; // .asciz   "^{BABugExample=^{BABugExample}}"
}
@property (copy) BABugExampleRedefinition *property;
@end

@implementation BABugExample
@synthesize property = _property;
@end

// CHECK-DWARF: private unnamed_addr constant [8 x i8] c"@16
// CHECK-MINGW: @".objc_sel_types_@16@0:8" = linkonce_odr hidden constant [8 x i8] c"@16@0:8\00"
// CHECK-MSVC: @".objc_sel_types_@16@0:8" = linkonce_odr hidden constant [8 x i8] c"@16@0:8\00"
// CHECK-ELF @".objc_sel_types_\0116\010:8" = linkonce_odr hidden constant [8 x i8] c"@16@0:8\00"

@class SCNCamera;
typedef SCNCamera C3DCamera;
typedef struct
{
    C3DCamera *presentationInstance;
}  C3DCameraStorage;

@interface SCNCamera
@end

@implementation SCNCamera
{
    C3DCameraStorage _storage;
}
@end
// CHECK-DWARF: private unnamed_addr constant [39 x i8] c"{?=\22presentationInstance\22@\22SCNCamera\22}\00"
// CHECK-MINGW: @"__objc_ivar_offset_SCNCamera._storage.{?\02@}"
// CHECK-MSVC: @"__objc_ivar_offset_SCNCamera._storage.{?\02@}"
// CHECK-ELF: @"__objc_ivar_offset_SCNCamera._storage.{?=\01}"

int i;
typeof(@encode(typeof(i))) e = @encode(typeof(i));
const char * Test(void)
{
    return e;
}
// CHECK-DWARF: @e ={{.*}} global [2 x i8] c"i\00", align 1
// CHECK-DWARF: define{{.*}} ptr @Test()
// CHECK-DWARF: ret ptr @e

// CHECK-MSVC: @e = dso_local global [2 x i8] c"i\00", align 1
// CHECK-MINGW: @e = dso_local global [2 x i8] c"i\00", align 1
// CHECK-ELF: @e = global [2 x i8] c"i\00", align 1
