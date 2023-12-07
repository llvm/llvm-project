// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSObject @end

@protocol DVTInvalidation
- (void)invalidate;  // expected-note {{method 'invalidate' declared here}}
@property int Prop; // expected-note {{property declared here}}
@end



@protocol DVTInvalidation;

@interface IBImageCatalogDocument : NSObject <DVTInvalidation>
@end

@implementation IBImageCatalogDocument // expected-warning {{auto property synthesis will not synthesize property 'Prop' declared in protocol 'DVTInvalidation'}} \
				       // expected-warning {{method 'invalidate' in protocol 'DVTInvalidation' not implemented}}
@end // expected-note {{add a '@synthesize' directive}}
