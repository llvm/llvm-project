// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -fobjc-gc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple i386-apple-darwin9 -fsyntax-only -fobjc-gc -verify -Wno-objc-root-class %s

struct S {
	__weak id w; // expected-warning {{__weak attribute cannot be specified on a field declaration}}
	__strong id p1;
};

@interface I
{
   __weak id w;	// OK
   __strong id LHS;
}
- (void) foo;
@end
@implementation I
- (void) foo { w = 0; LHS = w; }
@end

int main ()
{
	struct I {
        __weak id w1;  // expected-warning {{__weak attribute cannot be specified on a field declaration}}
	};
}
