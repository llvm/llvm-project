// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm  -fobjc-arc -o - %s | FileCheck %s

typedef const struct __CFString * CFStringRef;

@interface I 
@property CFStringRef P;
- (CFStringRef) CFMeth __attribute__((cf_returns_retained));
- (CFStringRef) newSomething;
- (CFStringRef) P __attribute__((cf_returns_retained));
@end

@implementation I
@synthesize P;
- (id) Meth {
    I* p1 = (id)[p1 P];
    id p2 = (id)[p1 CFMeth];
    id p3 = (id)[p1 newSomething];
    return (id) p1.P;
}
- (CFStringRef) CFMeth { return 0; }
- (CFStringRef) newSomething { return 0; }
- (CFStringRef) P { return 0; }
- (void) setP : (CFStringRef)arg {}
@end

CFStringRef SomeOtherFunc(void) __attribute__((cf_returns_retained));
id MMM(void)
{
  id obj = (id)((CFStringRef) __builtin___CFStringMakeConstantString ("" "Some CF String" ""));
  return 0;
}

// CHECK-NOT: call i8* @llvm.objc.retainAutoreleasedReturnValue
