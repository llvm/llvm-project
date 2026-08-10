// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct NSSize {
     		int width;
		struct {
		  int dim;
		} inner;
} NSSize;

typedef __attribute__((__ext_vector_type__(2))) float simd_float2;

@interface Foo  {
        NSSize _size;
}
@property NSSize size;
@property simd_float2 f2;
@end

void foo(void) { 
        Foo *f;
        f.size.width = 2.2; // expected-error {{expression is not assignable}}
	f.size.inner.dim = 200; // expected-error {{expression is not assignable}}
}

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

// clang used to crash compiling this code.
void test(Foo *f) {
  simd_float2 *v = &f.f2.xy; // expected-error {{cannot take the address of an rvalue}}
}
