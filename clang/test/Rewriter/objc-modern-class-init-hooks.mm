// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 
// rdar:// 11124354

@interface Root @end

@interface Super : Root
@end

@interface Sub : Super
@end

@implementation Sub @end

@implementation Root @end

@interface Root(Cat) @end

@interface Sub(Cat) @end

@implementation Root(Cat) @end

@implementation Sub(Cat) @end


// CHECK: #pragma section(".objc_inithooks$B", long, read, write)
// CHECK: __declspec(allocate(".objc_inithooks$B")) static void *OBJC_CLASS_SETUP[] = {
// CHECK:         (void *)&OBJC_CLASS_SETUP_$_Sub,
// CHECK:         (void *)&OBJC_CLASS_SETUP_$_Root,
// CHECK: };

// CHECK: #pragma section(".objc_inithooks$B", long, read, write)
// CHECK: __declspec(allocate(".objc_inithooks$B")) static void *OBJC_CATEGORY_SETUP[] = {
// CHECK:         (void *)&OBJC_CATEGORY_SETUP_$_Root_$_Cat,
// CHECK:         (void *)&OBJC_CATEGORY_SETUP_$_Sub_$_Cat,
// CHECK: };
