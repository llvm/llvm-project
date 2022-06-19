// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct NSSize {
     		int width;
		struct {
		  int dim;
		} inner;
} NSSize;

@interface Foo  {
        NSSize _size;
}
@property NSSize size;
@end

void foo(void) { 
        Foo *f;
        f.size.width = 2.2; // expected-error {{expression is not assignable}}
	f.size.inner.dim = 200; // expected-error {{expression is not assignable}}
}

// radar 7628953

@interface Gorf  {
}
- (NSSize)size;
@end

@implementation Gorf
- (void)MyView_sharedInit {
    self.size.width = 2.2; // expected-error {{expression is not assignable}}
}
- (NSSize)size {}
@end
