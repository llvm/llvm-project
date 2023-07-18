// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"id=struct objc_object *" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://11239894

extern "C" void *sel_registerName(const char *);

typedef struct objc_class * Class;

@interface Sub
- (void)dealloc;
@end

@interface I : Sub
- (void)dealloc;
@end

@implementation I
- (void)dealloc {
    return;
    [super dealloc];
}
@end

