// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -fobjc-runtime-has-weak -Wexplicit-ownership-type  -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -fobjc-runtime-has-weak -Wexplicit-ownership-type -verify -Wno-objc-root-class %s
// rdar://10244607

typedef const struct __CFString * CFStringRef;
@class NSString;

NSString *CFBridgingRelease(void);

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

// rdar://problem/10711456
__strong I *__strong test1; // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong test2); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong (test3)); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__unsafe_unretained __typeof__(test3) test4;
typedef __strong I *strong_I;
__unsafe_unretained strong_I test5;

// rdar://10907090
typedef void (^T) (void);
@interface NSObject @end
@protocol P;
@interface Radar10907090 @end

@implementation Radar10907090
- (void) MMM : (NSObject*) arg0 : (NSObject<P>**)arg : (id) arg1 : (id<P>*) arg2 {} // expected-warning {{method parameter of type 'NSObject<P> *__autoreleasing *' with no explicit ownership}} \
					// expected-warning {{method parameter of type '__autoreleasing id<P> *' with no explicit ownership}}
- (void) MM : (NSObject*) arg0 : (__strong NSObject**)arg : (id) arg1 : (__strong id*) arg2 {}
- (void) M : (NSObject**)arg0 : (id*)arg {} // expected-warning {{method parameter of type 'NSObject *__autoreleasing *' with no explicit ownership}} \
                                            // expected-warning {{method parameter of type '__autoreleasing id *' with no explicit ownership}}
- (void) N : (__strong NSObject***) arg0 : (__strong NSObject<P>***)arg : (float**) arg1 : (double) arg2 {} 
- (void) BLOCK : (T*) arg0 : (T)arg  : (__strong T*) arg1 {} // expected-warning-re {{method parameter of type '__autoreleasing T *' (aka 'void (^__autoreleasing *)({{(void)?}})') with no explicit ownership}}
@end

// rdar://12280826
@class NSMutableDictionary, NSError;
@interface Radar12280826
- (void)createInferiorTransportAndSetEnvironment:(NSMutableDictionary*)environment error:(__autoreleasing NSError**)error;
@end

@implementation Radar12280826
- (void)createInferiorTransportAndSetEnvironment:(NSMutableDictionary*)environment error:(__autoreleasing NSError**)error {}
@end

// <rdar://problem/12367446>
typedef __strong id strong_id;
typedef NSObject *NSObject_ptr;
typedef __strong NSObject *strong_NSObject_ptr;

// Warn
__strong id f1(void); // expected-warning{{ARC __strong lifetime qualifier on return type is ignored}}
NSObject __unsafe_unretained *f2(int); // expected-warning{{ARC __unsafe_unretained lifetime qualifier on return type is ignored}}
__autoreleasing NSObject *f3(void); // expected-warning{{ARC __autoreleasing lifetime qualifier on return type is ignored}}
NSObject * __strong f4(void); // expected-warning{{ARC __strong lifetime qualifier on return type is ignored}}
NSObject_ptr __strong f5(void); // expected-warning{{ARC __strong lifetime qualifier on return type is ignored}}

typedef __strong id (*fptr)(int); // expected-warning{{ARC __strong lifetime qualifier on return type is ignored}}

// Don't warn
strong_id f6(void);
strong_NSObject_ptr f7(void);
typedef __strong id (^block_ptr)(int);

// rdar://10127067
void test8_a(void) {
  __weak id *(^myBlock)(void);
  __weak id *var = myBlock();
  (void) (__strong id *) &myBlock;
  (void) (__weak id *) &myBlock; // expected-error {{cast}}
}
void test8_b(void) {
  __weak id (^myBlock)(void);
  (void) (__weak id *) &myBlock;
  (void) (__strong id *) &myBlock; // expected-error {{cast}}
}
void test8_c(void) {
  __weak id (^*(^myBlock)(void))(void);
  (void) (__weak id*) myBlock();
  (void) (__strong id*) myBlock(); // expected-error {{cast}}
  (void) (__weak id*) &myBlock; // expected-error {{cast}}
  (void) (__strong id*) &myBlock;
}

@class Test9;
void test9_a(void) {
  __weak Test9 **(^myBlock)(void);
  __weak Test9 **var = myBlock();
  (void) (__strong Test9 **) &myBlock;
  (void) (__weak Test9 **) &myBlock; // expected-error {{cast}}
}
void test9_b(void) {
  __weak Test9 *(^myBlock)(void);
  (void) (__weak Test9**) &myBlock;
  (void) (__strong Test9**) &myBlock; // expected-error {{cast}}
}
void test9_c(void) {
  __weak Test9 *(^*(^myBlock)(void))(void);
  (void) (__weak Test9 **) myBlock();
  (void) (__strong Test9 **) myBlock(); // expected-error {{cast}}
  (void) (__weak Test9 **) &myBlock; // expected-error {{cast}}
  (void) (__strong Test9 **) &myBlock;
}
