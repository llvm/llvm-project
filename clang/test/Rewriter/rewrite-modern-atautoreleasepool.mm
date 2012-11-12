// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o %t-rw.cpp
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// radar 11474836

extern "C"
void *sel_registerName(const char *);

@interface I
{
  id ivar;
}
- (id) Meth;
+ (id) MyAlloc;;
@end

@implementation I
- (id) Meth {
   @autoreleasepool {
      id p = [I MyAlloc];
      if (!p)
        return ivar;
   }
  return 0;
}
+ (id) MyAlloc {
    return 0;
}
@end

// CHECK: /* @autoreleasepool */ { __AtAutoreleasePool __autoreleasepool;
