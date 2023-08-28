// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface NSString @end

@interface NSObject @end

@interface SynthItAll
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthItAll
#if !__has_feature(objc_default_synthesize_properties)
@synthesize howMany, what;
#endif
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthSetter
#if !__has_feature(objc_default_synthesize_properties)
@synthesize howMany, what;
#endif

- (int) howMany {
    return self.howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return self.what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
#if !__has_feature(objc_default_synthesize_properties)
@synthesize howMany, what;
#endif

// - (int) howMany
- (void) setHowMany: (int) value {
    self.howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end


@interface SynthNone : NSObject
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthNone
#if !__has_feature(objc_default_synthesize_properties)
@synthesize howMany, what;  // REM: Redundant anyway
#endif

- (int) howMany {
    return self.howMany;
}
- (void) setHowMany: (int) value {
    self.howMany = value;
}

- (NSString*) what {
    return self.what;
}
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end

@protocol TopProtocol
  @property (readonly) id myString;
@end

@interface TopClass <TopProtocol> 
{
  id myString; 
}
@end

@interface SubClass : TopClass <TopProtocol>
@end

@implementation SubClass @end

@interface C @end
@interface C (Category)
@property int p; // expected-note 2 {{property declared here}}
@end
@implementation C (Category) // expected-warning {{property 'p' requires method 'p' to be defined}} \
                             // expected-warning {{property 'p' requires method 'setP:' to be defined}}
@end

// Don't complain if a property is already @synthesized by usr.
@interface D
{
}
@property int PROP;
@end

@implementation D
- (int) Meth { return self.PROP; }
#if __has_feature(objc_default_synthesize_properties)
@synthesize PROP=IVAR;
#endif
@end

@protocol MyProtocol 
@property (nonatomic, strong) NSString *requiredString; // expected-note {{property declared here}}

@optional
@property (nonatomic, strong) NSString *optionalString;
@end
 
@interface MyClass <MyProtocol> 
@end
 
@implementation MyClass // expected-warning {{auto property synthesis will not synthesize property 'requiredString' declared in protocol 'MyProtocol'}}
@end // expected-note {{add a '@synthesize' directive}}

@protocol NSObject @end
@protocol TMSourceManagerDelegate<NSObject>
@end

@protocol TMSourceManager <NSObject>
@property (nonatomic, assign) id <TMSourceManagerDelegate> delegate;
@end

@interface TMSourceManager
@property (nonatomic, assign) id <TMSourceManagerDelegate> delegate;
@end

@protocol TMTimeZoneManager <TMSourceManager>
@end

@interface TimeZoneManager : TMSourceManager <TMTimeZoneManager>
@end

@implementation TimeZoneManager
@end

@protocol BaseProt
@property (assign) id prot;
@end

@interface Base<BaseProt>
@end

@interface I : Base<BaseProt>
@end

@implementation I
@end
