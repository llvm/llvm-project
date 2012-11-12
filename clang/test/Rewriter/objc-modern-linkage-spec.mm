// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-attributes -D"__declspec(X)=" %t-rw.cpp
// rdar://11169733

extern "C" __declspec(dllexport)
@interface Test @end

@implementation Test @end

extern "C" {
__declspec(dllexport)
@interface Test1 @end

@implementation Test1 @end

__declspec(dllexport)
@interface Test2 @end

@implementation Test2 @end
};

