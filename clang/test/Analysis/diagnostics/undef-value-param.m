// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx -analyzer-output=plist-multi-file %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/undef-value-param.m.plist -

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end
typedef const void * CFTypeRef;
extern void CFRelease(CFTypeRef cf);

@interface Cell : NSObject
- (void)test;
@end

@interface SpecialString
+ (id)alloc;
- (oneway void)release;
@end

typedef SpecialString* SCDynamicStoreRef;
static void CreateRef(SCDynamicStoreRef *storeRef, unsigned x);
static void CreateRefUndef(SCDynamicStoreRef *storeRef, unsigned x);
SCDynamicStoreRef anotherCreateRef(unsigned *err, unsigned x);

@implementation Cell
- (void) test {
    SCDynamicStoreRef storeRef = 0;
    CreateRef(&storeRef, 4); 
                             //expected-note@-1{{Calling 'CreateRef'}}
                             //expected-note@-2{{Returning from 'CreateRef'}}
    CFRelease(storeRef); //expected-warning {{Null pointer argument in call to CFRelease}}
                         //expected-note@-1{{Null pointer argument in call to CFRelease}}
}

- (void)test2 {
    SCDynamicStoreRef storeRef; // expected-note {{'storeRef' declared without an initial value}}
    CreateRefUndef(&storeRef, 4);
                             //expected-note@-1{{Calling 'CreateRefUndef'}}
                             //expected-note@-2{{Returning from 'CreateRefUndef'}}
    CFRelease(storeRef); //expected-warning {{1st function call argument is an uninitialized value}}
                         //expected-note@-1{{1st function call argument is an uninitialized value}}
}
@end

static void CreateRef(SCDynamicStoreRef *storeRef, unsigned x) {
    unsigned err = 0;
    SCDynamicStoreRef ref = anotherCreateRef(&err, x);
    if (err) { 
               //expected-note@-1{{Assuming 'err' is not equal to 0}}
               //expected-note@-2{{Taking true branch}}
        CFRelease(ref);
        ref = 0; // expected-note{{nil object reference stored to 'ref'}}
    }
    *storeRef = ref; // expected-note{{nil object reference stored to 'storeRef'}}
}

static void CreateRefUndef(SCDynamicStoreRef *storeRef, unsigned x) {
  unsigned err = 0;
  SCDynamicStoreRef ref = anotherCreateRef(&err, x);
  if (err) {
             //expected-note@-1{{Assuming 'err' is not equal to 0}}
             //expected-note@-2{{Taking true branch}}
    CFRelease(ref);
    return; // expected-note{{Returning without writing to '*storeRef'}}
  }
  *storeRef = ref;
}

