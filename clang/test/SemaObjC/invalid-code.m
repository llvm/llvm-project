// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions -Wno-objc-root-class %s

// rdar://6124613
void test1() {
  void *xyzzy = 0;
  void *p = @xyzzy; // expected-error {{unexpected '@' in program}}
}

// <rdar://problem/7495713>
// This previously triggered a crash because the class has not been defined.
@implementation RDar7495713 (rdar_7495713_cat)  // expected-error{{cannot find interface declaration for 'RDar7495713'}}
- (id) rdar_7495713 {
  __PRETTY_FUNCTION__; // expected-warning{{expression result unused}}
}
@end

// <rdar://problem/7881045>
// This previously triggered a crash because a ';' was expected after the @throw statement.
void foo() {
  @throw (id)0 // expected-error{{expected ';' after @throw}}
}

// <rdar://problem/10415026>
@class NSView;
@implementation IBFillView(IBFillViewIntegration) // expected-error {{cannot find interface declaration for 'IBFillView'}}
- (NSView *)ibDesignableContentView {
    [Cake lie]; // expected-error {{undeclared}}
    return self;
}
@end

@interface I
@end
@interface I2
@end

@implementation I // expected-note {{started here}}
-(void) foo {}

@implementation I2 // expected-error {{missing '@end'}}
-(void) foo2 {}
@end

@end // expected-error {{'@end' must appear in an Objective-C context}}

@class ForwardBase;
@implementation SomeI : ForwardBase // expected-error {{cannot find interface declaration for 'ForwardBase', superclass of 'SomeI'}} \
                                    // expected-warning {{cannot find interface declaration for 'SomeI'}}
-(void)meth {}
@end

@interface I3
__attribute__((unavailable)) @interface I4 @end // expected-error {{Objective-C declarations may only appear in global scope}}
@end
