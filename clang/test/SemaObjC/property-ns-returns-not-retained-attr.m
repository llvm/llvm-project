// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://9636091

@interface I
@property (nonatomic, retain) id newName __attribute__((ns_returns_not_retained)) ;

@property (nonatomic, retain) id newName1 __attribute__((ns_returns_not_retained)) ;
- (id) newName1 __attribute__((ns_returns_not_retained));

@property (nonatomic, retain) id newName2 __attribute__((ns_returns_not_retained)); // expected-note {{roperty declared here}}
- (id) newName2;   // expected-warning {{property declared as returning non-retained objects; getter returning retained objects}}
@end

@implementation I
@synthesize newName;

@synthesize newName1;
- (id) newName1 { return 0; }

@synthesize newName2;
@end
