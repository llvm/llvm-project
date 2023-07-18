// RUN: %clang_cc1 -fsyntax-only -Wobjc-missing-property-synthesis -verify -Wno-objc-root-class -triple=x86_64-apple-macos10.10 %s
// rdar://11295716

@interface NSObject 
- (void) release;
- (id) retain;
@end
@class NSString;

@interface SynthItAll : NSObject
@property int howMany; // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@property (retain) NSString* what; // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@end

@implementation SynthItAll // expected-note 2 {{detected while default synthesizing properties in class implementation}}
//@synthesize howMany, what;
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany;   // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@property (nonatomic, retain) NSString* what;  // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@end

@implementation SynthSetter // expected-note 2 {{detected while default synthesizing properties in class implementation}}
//@synthesize howMany, what;

- (int) howMany {
    return _howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return _what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany; // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}} 
@property (nonatomic, retain) NSString* what; // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@end

@implementation SynthGetter // expected-note 2 {{detected while default synthesizing properties in class implementation}}
//@synthesize howMany, what;

// - (int) howMany
- (void) setHowMany: (int) value {
    _howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (_what != value) {
        [_what release];
        _what = [value retain];
    }
}
@end


@interface SynthNone : NSObject
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthNone
//@synthesize howMany, what;  // REM: Redundant anyway

- (int) howMany {
    return howMany; // expected-error {{use of undeclared identifier 'howMany'}}
}
- (void) setHowMany: (int) value {
    howMany = value; // expected-error {{use of undeclared identifier 'howMany'}}
}

- (NSString*) what {
    return what; // expected-error {{use of undeclared identifier 'what'}}
}
- (void) setWhat: (NSString*) value {
    if (what != value) { // expected-error {{use of undeclared identifier 'what'}}
        [what release]; // expected-error {{use of undeclared identifier 'what'}}
        what = [value retain]; // expected-error {{use of undeclared identifier 'what'}}
    }
}
@end

// rdar://8349319
// No default synthesis if implementation has getter (readonly) and setter(readwrite) methods.
@interface DSATextSearchResult 
@property(assign,readonly) float relevance;
@property(assign,readonly) char isTitleMatch;
@end

@interface DSANodeSearchResult : DSATextSearchResult {}
@end


@implementation DSATextSearchResult 
-(char)isTitleMatch {
    return (char)0;
}

-(float)relevance {
    return 0.0;
}
@end

@implementation DSANodeSearchResult
-(id)initWithNode:(id )node relevance:(float)relevance isTitleMatch:(char)isTitleMatch {
        relevance = 0.0;        
        isTitleMatch = 'a';
	return self;
}
@end

@interface rdar11333367
@property enum A x; // expected-note {{forward declaration of 'enum A'}} expected-note {{property declared here}}
@property struct B y; // expected-note {{forward declaration of 'struct B'}} expected-note {{property declared here}} \
                      // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}
@end
@implementation rdar11333367 // expected-error {{cannot synthesize property 'y' with incomplete type 'struct B'}} \
                             // expected-note {{detected while default synthesizing properties in class implementation}}
@synthesize x; // expected-error {{cannot synthesize property 'x' with incomplete type 'enum A'}}
@end

// rdar://17774815
@interface ZXParsedResult
@property (nonatomic, copy, readonly) NSString *description; // expected-note {{property declared here}}
@end

@interface ZXCalendarParsedResult : ZXParsedResult

@property (nonatomic, copy, readonly) NSString *description; // expected-warning {{auto property synthesis will not synthesize property 'description'; it will be implemented by its superclass}}

@end

@implementation ZXCalendarParsedResult // expected-note {{detected while default synthesizing properties in class implementation}}
- (NSString *) Meth {
    return _description; // expected-error {{use of undeclared identifier '_description'}}
}
@end

@interface DontWarnOnUnavailable

// No warning expected:
@property (nonatomic, readonly) int un1 __attribute__((unavailable));
@property (readwrite) int un2 __attribute__((availability(macos, unavailable)));

@property (readwrite) int un3 __attribute__((availability(ios, unavailable))); // expected-warning {{auto property synthesis is synthesizing property not explicitly synthesized}}

@end

@implementation DontWarnOnUnavailable // expected-note {{detected while default synthesizing properties in class implementation}}

@end
