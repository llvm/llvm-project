// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-10.7 %t.mm -o %t-rw.cpp
// RUN: FileCheck --input-file=%t-rw.cpp %s

@interface I @end
@implementation I @end

// CHECK: __OBJC_RW_DLLIMPORT struct objc_class *objc_getClass(const char *);
// CHECK: __OBJC_RW_DLLIMPORT struct objc_class *objc_getMetaClass(const char *);

