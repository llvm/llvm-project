// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// radar 8608293

typedef unsigned long size_t;
void *sel_registerName(const char *);

extern "C" void nowarn(id);

extern "C" void noblockwarn(void (^)());

@interface INTFOFPROP 
@property (readwrite, retain) INTFOFPROP *outer;
@property (readwrite, retain) id inner;
@end

@interface NSSet
- (NSSet *)objectsPassingTest:(char (^)(id obj, char *stop))predicate ;
@end

@interface INTF
- (NSSet *)Meth;
@end

@implementation INTF

- (NSSet *)Meth
{
    NSSet *aces;

    noblockwarn(^() {
        INTFOFPROP *ace;
        nowarn(ace.outer.inner);
        noblockwarn(^() {
          INTFOFPROP *ace;
          nowarn(ace.outer.inner);
        });
    });

    noblockwarn(^() {
        INTFOFPROP *ace;
        nowarn(ace.outer.inner);
    });

return [aces objectsPassingTest:^(id obj, char *stop)
    {
        INTFOFPROP *ace = (INTFOFPROP *)obj;
        nowarn(ace.outer.inner);
        return (char)0;
    }];

}
@end
