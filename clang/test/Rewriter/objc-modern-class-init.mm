// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

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
