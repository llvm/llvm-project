// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -Wexplicit-ownership-type -verify -Wno-objc-root-class %s

typedef const struct __CFString * CFStringRef;
@class NSString;

NSString *CFBridgingRelease();

typedef NSString * PNSString;

typedef __autoreleasing NSString * AUTORELEASEPNSString;

@interface I @end

@implementation I
- (CFStringRef)myString
{
    CFStringRef myString =
      (__bridge CFStringRef) (__strong NSString *)CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}

    myString =
      (__bridge CFStringRef) (__autoreleasing PNSString) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}
    myString =
      (__bridge CFStringRef) (AUTORELEASEPNSString) CFBridgingRelease(); // OK
    myString =
      (__bridge CFStringRef) (typeof(__strong NSString *)) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}
    return myString;
}

- (void)decodeValueOfObjCType:(const char *)type at:(void *)addr {
        __autoreleasing id *stuff = (__autoreleasing id *)addr;
}
@end

__strong I *__strong test1; // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong test2); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong (test3)); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__unsafe_unretained __typeof__(test3) test4;
typedef __strong I *strong_I;
__unsafe_unretained strong_I test5;

typedef void (^T) ();
@interface NSObject @end
@protocol P;
@interface Radar10907090 @end

@implementation Radar10907090
- (void) MMM : (NSObject*) arg0 : (NSObject<P>*&)arg : (id) arg1 : (id<P>&) arg2 {} // expected-warning {{method parameter of type 'NSObject<P> *__autoreleasing &' with no explicit ownership}} \
					// expected-warning {{method parameter of type '__autoreleasing id<P> &' with no explicit ownership}}
- (void) MM : (NSObject*) arg0 : (__strong NSObject**)arg : (id) arg1 : (__strong id*) arg2 {}
- (void) M : (NSObject**)arg0 : (id*)arg {} // expected-warning {{method parameter of type 'NSObject *__autoreleasing *' with no explicit ownership}} \
                                            // expected-warning {{method parameter of type '__autoreleasing id *' with no explicit ownership}}
- (void) N : (__strong NSObject***) arg0 : (__strong NSObject<P>***)arg : (float**) arg1 : (double) arg2 {} 
- (void) BLOCK : (T&) arg0 : (T)arg  : (__strong T*) arg1 {} // expected-warning {{method parameter of type '__autoreleasing T &' (aka 'void (^__autoreleasing &)()') with no explicit ownership}}
@end

@class NSMutableDictionary, NSError;
@interface Radar12280826
- (void)createInferiorTransportAndSetEnvironment:(NSMutableDictionary*)environment error:(__autoreleasing NSError*&)error;
@end

@implementation Radar12280826
- (void)createInferiorTransportAndSetEnvironment:(NSMutableDictionary*)environment error:(__autoreleasing NSError*&)error {}
@end

