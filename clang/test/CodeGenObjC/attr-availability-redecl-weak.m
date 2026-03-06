// RUN: %clang_cc1 -triple arm64-apple-ios12.0 -emit-llvm -o - %s | FileCheck %s

// Test that isWeakImported() correctly traverses the redeclaration chain
// to find availability attributes, even when a forward declaration (@class)
// without availability attributes becomes the most recent declaration.

// Case 1: @interface (with availability) first, then @class (without availability).
// The @class becomes getMostRecentDecl(). Without the fix, isWeakImported()
// would only check the @class's attributes and miss the availability attribute,
// resulting in strong linkage instead of extern_weak.

__attribute__((availability(ios,introduced=14.0)))
@interface WeakRedecl1
@end

@class WeakRedecl1;

@implementation WeakRedecl1 (TestCategory1)
@end

// CHECK: @"OBJC_CLASS_$_WeakRedecl1" = extern_weak global

// Case 2: @class first, then @interface (with availability).
// This order already worked before the fix because @interface becomes
// getMostRecentDecl() and carries the availability attribute.
// We test it here to ensure the fix doesn't regress this case.

@class WeakRedecl2;

__attribute__((availability(ios,introduced=14.0)))
@interface WeakRedecl2
@end

@implementation WeakRedecl2 (TestCategory2)
@end

// CHECK: @"OBJC_CLASS_$_WeakRedecl2" = extern_weak global

// Case 3: Single declaration with availability (baseline, no redeclaration).
__attribute__((availability(ios,introduced=14.0)))
@interface WeakSingle
@end

@implementation WeakSingle (TestCategory3)
@end

// CHECK: @"OBJC_CLASS_$_WeakSingle" = extern_weak global
