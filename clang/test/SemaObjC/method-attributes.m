// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s

@class NSString;

@interface A
-t1 __attribute__((noreturn));
- (NSString *)stringByAppendingFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
-(void) m0 __attribute__((noreturn));
-(void) m1 __attribute__((unused));
@end


@interface INTF
- (int) foo1: (int)arg1 __attribute__((deprecated));

- (int) foo: (int)arg1;  // expected-note {{method 'foo:' declared here}}

- (int) foo2: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{method 'foo2:' declared here}}
- (int) foo3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) __attribute__((ns_consumes_self));
@end

@implementation INTF
- (int) foo: (int)arg1  __attribute__((deprecated)){ // expected-warning {{attributes on method implementation and its declaration must match}}
        return 10;
}
- (int) foo1: (int)arg1 {
        return 10;
}
- (int) foo2: (int)arg1 __attribute__((deprecated)) {  // expected-warning {{attributes on method implementation and its declaration must match}}
        return 10;
}
- (int) foo3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) __attribute__((ns_consumes_self)) {return 0; }
- (void) dep __attribute__((deprecated)) { } // OK private methodn
@end


// rdar://10529259
#define IBAction void)__attribute__((ibaction)

@interface Foo 
- (void)doSomething1:(id)sender;
- (void)doSomething2:(id)sender; // expected-note {{method 'doSomething2:' declared here}}
@end

@implementation Foo
- (void)doSomething1:(id)sender{}
- (void)doSomething2:(id)sender{}
@end

@interface Bar : Foo
- (IBAction)doSomething1:(id)sender;
@end
@implementation Bar
- (IBAction)doSomething1:(id)sender {}
- (IBAction)doSomething2:(id)sender {} // expected-warning {{attributes on method implementation and its declaration must match}}
- (IBAction)doSomething3:(id)sender {}
@end

// rdar://11593375
@interface NSObject @end

@interface Test : NSObject
-(id)method __attribute__((deprecated));
-(id)method1;
-(id)method2 __attribute__((aligned(16)));
- (id) method3: (int)arg1 __attribute__((aligned(16)))  __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{method 'method3:' declared here}}
- (id) method4: (int)arg1 __attribute__((aligned(16)))  __attribute__((deprecated)) __attribute__((unavailable)); 
@end

@implementation Test
-(id)method __attribute__((aligned(16))) __attribute__((aligned(16))) __attribute__((deprecated)) {
    return self;
}
-(id)method1 __attribute__((aligned(16))) {
    return self;
}
-(id)method2 {
    return self;
}
- (id) method3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) {  // expected-warning {{attributes on method implementation and its declaration must match}}
        return self;
}
- (id) method4: (int)arg1 __attribute__((aligned(16))) __attribute__((deprecated)) __attribute__((unavailable)) {
  return self;
}
@end
