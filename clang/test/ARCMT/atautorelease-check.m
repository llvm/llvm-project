// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 %s

#if __has_feature(objc_arr)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

typedef struct _NSZone NSZone;
typedef int BOOL;
typedef unsigned NSUInteger;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;

- (NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)allocWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;

+ (id)copyWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
+ (id)mutableCopyWithZone:(NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

extern void NSRecycleZone(NSZone *zone);

NS_AUTOMATED_REFCOUNT_UNAVAILABLE
@interface NSAutoreleasePool : NSObject { // expected-note 13 {{marked unavailable here}}
@private
    void    *_token;
    void    *_reserved3;
    void    *_reserved2;
    void    *_reserved;
}

+ (void)addObject:(id)anObject;

- (void)addObject:(id)anObject;

- (void)drain;

@end


void NSLog(id, ...);

int main (int argc, const char * argv[]) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init]; 
    NSAutoreleasePool *chunkPool = [[NSAutoreleasePool alloc] init]; // expected-error 2 {{'NSAutoreleasePool' is unavailable}}

    while (argc) {
      [chunkPool release];
      // the following pool was not released in this scope, don't touch it. 
      chunkPool = [[NSAutoreleasePool alloc] init]; // expected-error {{'NSAutoreleasePool' is unavailable}}
    }

    [chunkPool drain];
    [pool drain];

    return 0;
}

void f(void) {
    NSAutoreleasePool * pool;  // expected-error {{'NSAutoreleasePool' is unavailable}}

    for (int i=0; i != 10; ++i) {
      id x = pool; // We won't touch a NSAutoreleasePool if we can't safely
                   // remove all the references to it.
    }

    pool = [[NSAutoreleasePool alloc] init];  // expected-error {{'NSAutoreleasePool' is unavailable}}
    NSLog(@"%s", "YES");
    [pool release];
}

void f2(void) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init]; // expected-error 2 {{'NSAutoreleasePool' is unavailable}} \
                                            // expected-note {{scope begins here}}

    // 'x' is declared inside the "pool scope" but used outside it, if we create
    // a @autorelease scope it will be undefined outside it so don't touch the pool.
    int x = 0; // expected-note {{declared here}}

    [pool release]; // expected-note {{scope ends here}}
    
    ++x; // expected-error {{a name is referenced outside the NSAutoreleasePool scope that it was declared in}}
}

void f3(void) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init]; // expected-error 2 {{'NSAutoreleasePool' is unavailable}} \
                                            // expected-note {{scope begins here}}

    struct S { int x; }; // expected-note {{declared here}}

    [pool release]; // expected-note {{scope ends here}}

    struct S *var; // expected-error {{a name is referenced outside the NSAutoreleasePool scope that it was declared in}}
    var->x = 0;
}

void f4(void) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init]; // expected-error 2 {{'NSAutoreleasePool' is unavailable}} \
                                            // expected-note {{scope begins here}}

    enum { Bar }; // expected-note {{declared here}}

    [pool release]; // expected-note {{scope ends here}}

    int x = Bar; // expected-error {{a name is referenced outside the NSAutoreleasePool scope that it was declared in}}
}

void f5(void) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init]; // expected-error 2 {{'NSAutoreleasePool' is unavailable}} \
                                            // expected-note {{scope begins here}}

    typedef int Bar; // expected-note {{declared here}}

    [pool release]; // expected-note {{scope ends here}}

    Bar x; // expected-error {{a name is referenced outside the NSAutoreleasePool scope that it was declared in}}
}
