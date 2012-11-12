// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D_Bool=bool -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D_Bool=bool -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// radar 8143056

typedef struct objc_class *Class;
typedef unsigned NSPointerFunctionsOptions;
extern "C" id NSClassFromObject(id object);
void *sel_registerName(const char *);

struct NSSlice {
  int i1;
};

@interface NSConcretePointerFunctions {
  @public
    struct NSSlice slice;
}
- (bool)initializeSlice:(struct NSSlice *)slicep withOptions:(NSPointerFunctionsOptions)options;
@end

@implementation NSConcretePointerFunctions
- (id)initWithOptions:(NSPointerFunctionsOptions)options {
      if (![NSClassFromObject(self) initializeSlice:&slice withOptions:options])
        return 0;
      return self;
  }
- (bool)initializeSlice:(struct NSSlice *)slicep withOptions:(NSPointerFunctionsOptions)options {
    return 0;
  }
@end

@interface I1 @end

@implementation I1
+ (struct s1 *) f0 {
  return 0;
}
@end
