// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"id=struct objc_object*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"id=struct objc_object*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// rdar:// 9878420

typedef unsigned long size_t;

void objc_enumerationMutation(id);
void *sel_registerName(const char *);
typedef void (^CoreDAVCompletionBlock)(void);

@interface I
- (void)M;
- (id) ARR;
@property (readwrite, copy, nonatomic) CoreDAVCompletionBlock c;
@end

@implementation I
- (void)M {
    I* ace;
    self.c = ^() {
          // Basic correctness check for the changes.
	  [ace ARR];
          for (I *privilege in [ace ARR]) { }
    };
    self.c = ^() {
          // Basic correctness test for the changes.
	  [ace ARR];
    };
}
@end
