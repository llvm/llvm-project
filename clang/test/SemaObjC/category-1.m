// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface MyClass1 @end

@protocol p1,p2,p3; // expected-note {{protocol 'p1' has no definition}} \
                    // expected-note {{protocol 'p2' has no definition}}

@interface MyClass1 (Category1)  <p1> // expected-warning {{cannot find protocol definition for 'p1'}} expected-note {{previous definition is here}}
@end

@interface MyClass1 (Category1)  // expected-warning {{duplicate definition of category 'Category1' on interface 'MyClass1'}}
@end

@interface MyClass1 (Category3) 
@end

@interface MyClass1 (Category4) @end // expected-note {{previous definition is here}}
@interface MyClass1 (Category5) @end
@interface MyClass1 (Category6) @end
@interface MyClass1 (Category7) @end // expected-note {{previous definition is here}}
@interface MyClass1 (Category8) @end // expected-note {{previous definition is here}}


@interface MyClass1 (Category4) @end // expected-warning {{duplicate definition of category 'Category4' on interface 'MyClass1'}}
@interface MyClass1 (Category7) @end // expected-warning {{duplicate definition of category 'Category7' on interface 'MyClass1'}}
@interface MyClass1 (Category8) @end // expected-warning {{duplicate definition of category 'Category8' on interface 'MyClass1'}}


@protocol p3 @end

@interface MyClass1 (Category) <p2, p3> @end  // expected-warning {{cannot find protocol definition for 'p2'}}

@interface UnknownClass  (Category) @end // expected-error {{cannot find interface declaration for 'UnknownClass'}}

@class MyClass2; // expected-note{{forward declaration of class here}}

@interface MyClass2  (Category) @end  // expected-error {{cannot define category for undefined class 'MyClass2'}}

@interface XCRemoteComputerManager
@end

@interface XCRemoteComputerManager() 
@end 

@interface XCRemoteComputerManager()
@end

@interface XCRemoteComputerManager(x) // expected-note {{previous definition is here}}
@end 

@interface XCRemoteComputerManager(x) // expected-warning {{duplicate definition of category 'x' on interface 'XCRemoteComputerManager'}}
@end

@implementation XCRemoteComputerManager
@end

@implementation XCRemoteComputerManager(x) // expected-note {{previous definition is here}}
@end

@implementation XCRemoteComputerManager(x) // expected-error {{reimplementation of category 'x' for class 'XCRemoteComputerManager'}}
@end

// <rdar://problem/7249233>

@protocol MultipleCat_P
-(void) im0; // expected-note {{method 'im0' declared here}}
@end

@interface MultipleCat_I @end

@interface MultipleCat_I()  @end

@interface MultipleCat_I() <MultipleCat_P>  @end

@implementation MultipleCat_I // expected-warning {{method 'im0' in protocol 'MultipleCat_P' not implemented}}
@end

// <rdar://problem/7680391> - Handle nameless categories with no name that refer
// to an undefined class
@interface RDar7680391 () @end // expected-error{{cannot find interface declaration}}

// <rdar://problem/8891119> - Handle @synthesize being used in conjunction
// with explicitly declared accessor.
@interface RDar8891119 {
  id _name;
}
@end
@interface RDar8891119 ()
- (id)name;
@end
@interface RDar8891119 ()
@property (copy) id name;
@end
@implementation RDar8891119
@synthesize name = _name;
@end

// rdar://10968158
@class I; // expected-note {{forward declaration}}
@implementation I(cat) // expected-error{{cannot find interface declaration}}
@end

// <rdar://problem/11478173>
@interface Unrelated
- foo;
@end

@interface Blah (Blarg) // expected-error{{cannot find interface declaration for 'Blah'}}
- foo;
@end
