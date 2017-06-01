// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s

#if __has_attribute(objc_requires_property_definitions)
__attribute ((objc_requires_property_definitions)) 
#endif
@interface NoAuto // expected-note 2 {{class with specified objc_requires_property_definitions attribute is declared here}}
@property int NoAutoProp; // expected-note 2 {{property declared here}}
@end

@implementation NoAuto  // expected-warning {{property 'NoAutoProp' requires method 'NoAutoProp' to be defined}} \
                        // expected-warning {{property 'NoAutoProp' requires method 'setNoAutoProp:'}}
@end

__attribute ((objc_requires_property_definitions))  // redundant, just for testing
@interface Sub : NoAuto  // expected-note 3 {{class with specified objc_requires_property_definitions attribute is declared here}}
@property (copy) id SubProperty; // expected-note 2 {{property declared here}}
@end

@implementation Sub // expected-warning {{property 'SubProperty' requires method 'SubProperty' to be defined}} \
                    // expected-warning {{property 'SubProperty' requires method 'setSubProperty:' to be defined}}
@end

@interface Deep : Sub
@property (copy) id DeepProperty;
@property (copy) id DeepSynthProperty;
@property (copy) id DeepMustSynthProperty; // expected-note {{property declared here}}
@end

@implementation Deep // expected-warning {{property 'DeepMustSynthProperty' requires method 'setDeepMustSynthProperty:' to be defined}}
@dynamic DeepProperty;
@synthesize DeepSynthProperty;
- (id) DeepMustSynthProperty { return 0; }
@end

__attribute ((objc_requires_property_definitions)) // expected-error {{'objc_requires_property_definitions' attribute only applies to Objective-C interfaces}}
@interface Deep(CAT)
@end

__attribute ((objc_requires_property_definitions)) // expected-error {{'objc_requires_property_definitions' attribute only applies to Objective-C interfaces}}
@protocol P @end

// rdar://13388503
@interface NSObject @end
@protocol Foo
@property (readonly) char isFoo; // expected-note {{property declared here}}
@property (readonly) char isNotFree;  // expected-note {{property declared here}}
@end

@interface Bar : NSObject <Foo>
@end

@implementation Bar
- (char)isFoo {
    return 0;
}
- (char)isNotFree {
    return 0;
}
@end

@interface Baz : Bar
@end

@interface Baz ()
@property (readwrite) char isFoo; // expected-warning {{auto property synthesis will not synthesize property 'isFoo' because it is 'readwrite' but it will be synthesized 'readonly' via another property}}
@property char Property1; // expected-warning {{auto property synthesis will not synthesize property 'Property1' because it cannot share an ivar with another synthesized property}}
@property char Property2;
@property (readwrite) char isNotFree; // expected-warning {{auto property synthesis will not synthesize property 'isNotFree'}}
@end

@implementation Baz { // expected-note {{detected while default synthesizing properties in class implementation}}
    char _isFoo;
    char _isNotFree;
}
@synthesize Property2 = Property1; // expected-note {{property synthesized here}}

- (void) setIsNotFree : (char)Arg {
  _isNotFree = Arg;
}

@end

// More test where such warnings should not be issued.
@protocol MyProtocol
-(void)setProp1:(id)x;
@end

@protocol P1 <MyProtocol>
@end

@interface B
@property (readonly) id prop;  // expected-note {{property declared here}}
@property (readonly) id prop1;  // expected-note {{property declared here}}
@property (readonly) id prop2;  // expected-note {{property declared here}}
@end

@interface B()
-(void)setProp:(id)x;
@end

@interface B(cat)
@property (readwrite) id prop2;
@end

@interface S : B<P1>
@property (assign,readwrite) id prop; // expected-warning {{auto property synthesis will not synthesize property 'prop'}}
@property (assign,readwrite) id prop1; // expected-warning {{auto property synthesis will not synthesize property 'prop1'}}
@property (assign,readwrite) id prop2; // expected-warning {{auto property synthesis will not synthesize property 'prop2'}}
@end

@implementation S // expected-note 3 {{detected while default synthesizing properties in class implementation}}
@end

// rdar://14085456
// No warning must be issued in this test.
@interface ParentObject
@end

@protocol TestObject 
@property (readonly) int six;
@end

@interface TestObject : ParentObject <TestObject>
@property int six;
@end

@implementation TestObject
@synthesize six;
@end

// rdar://14094682
// no warning in this test
@interface ISAChallenge : NSObject {
}

@property (assign, readonly) int failureCount;
@end

@interface ISSAChallenge : ISAChallenge {
    int _failureCount;
}
@property (assign, readwrite) int failureCount;
@end

@implementation ISAChallenge
- (int)failureCount {
    return 0;
}
@end

@implementation ISSAChallenge

@synthesize failureCount = _failureCount;
@end

__attribute ((objc_requires_property_definitions(1))) // expected-error {{'objc_requires_property_definitions' attribute takes no arguments}}
@interface I1
@end

// rdar://15051465
@protocol SubFooling
  @property(nonatomic, readonly) id hoho; // expected-note 2 {{property declared here}}
@end

@protocol Fooing<SubFooling>
  @property(nonatomic, readonly) id muahahaha; // expected-note 2 {{property declared here}}
@end

typedef NSObject<Fooing> FooObject;

@interface Okay : NSObject<Fooing>
@end

@implementation Okay // expected-warning {{auto property synthesis will not synthesize property 'muahahaha' declared in protocol 'Fooing'}} expected-warning {{auto property synthesis will not synthesize property 'hoho' declared in protocol 'SubFooling'}}
@end

@interface Fail : FooObject
@end

@implementation Fail // expected-warning {{auto property synthesis will not synthesize property 'muahahaha' declared in protocol 'Fooing'}} expected-warning {{auto property synthesis will not synthesize property 'hoho' declared in protocol 'SubFooling'}}
@end

// rdar://16089191
@class NSURL;

@interface Root
- (void)setFileURL : (NSURL *) arg;
- (void)setFile : (NSURL *) arg;
- (NSURL *)fileSys;
- (void)setFileSys : (NSURL *) arg;
- (NSURL *)fileKerl;
@end

@interface SuperClass : Root
- (NSURL *)fileURL;
- (NSURL *)file;
- (NSURL *)fileLog;
- (void)setFileLog : (NSURL *) arg;
- (void)setFileKerl : (NSURL *) arg;
@end

@protocol r16089191Protocol
@property (readonly) NSURL *fileURL;
@property (copy) NSURL *file;
@property (copy) NSURL *fileSys;
@property (copy) NSURL *fileLog;
@property (copy) NSURL *fileKerl;
@end

@interface SubClass : SuperClass <r16089191Protocol>
@end

@implementation SubClass
@end
