// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://13075400

@protocol FooAsID
@property (assign) id foo; // expected-note 2 {{it could also be property of type 'id' declared here}} \\
			 // expected-warning {{property of type 'id' was selected for synthesis}}
@end

@protocol FooAsDouble
@property double foo; // expected-warning 2 {{property of type 'double' was selected for synthesis}} \
		      // expected-note {{it could also be property of type 'double' declared here}}
@end

@protocol FooAsShort
@property short foo; // expected-note {{it could also be property of type 'short' declared here}}
@end

@interface NSObject @end

@interface AnObject : NSObject<FooAsDouble,FooAsID>
@end

@interface Sub : AnObject
@end

@implementation Sub
@synthesize foo=_MyFooIvar; // expected-note {{property synthesized here}}
@end


@interface AnotherObject : NSObject<FooAsDouble, FooAsID,FooAsDouble, FooAsID, FooAsDouble,FooAsID>
@end

@implementation AnotherObject
@synthesize foo=_MyFooIvar; // expected-note {{property synthesized here}}
@end


@interface YetAnotherObject : NSObject<FooAsID,FooAsShort, FooAsDouble,FooAsID, FooAsShort>
@end

@implementation YetAnotherObject
@synthesize foo=_MyFooIvar; // expected-note {{property synthesized here}}
@end

double func(YetAnotherObject *object) {
  return [object foo]; // expected-error {{returning 'id' from a function with incompatible result type 'double'}}
}
