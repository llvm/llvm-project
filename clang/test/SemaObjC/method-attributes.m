// RUN: %clang_cc1 -triple i386-unknown-unknown -verify -fsyntax-only -Wno-objc-root-class %s

@class NSString;

@interface A
-t1 __attribute__((noreturn));
- (NSString *)stringByAppendingFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
-(void) m0 __attribute__((noreturn));
-(void) m1 __attribute__((unused));
-(void) m2 __attribute__((stdcall));
-(void) m3 __attribute__((optnone));
@end


@interface INTF
- (int) foo1: (int)arg1 __attribute__((deprecated));

- (int) foo: (int)arg1;

- (int) foo2: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable));
- (int) foo3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) __attribute__((ns_consumes_self));
@end

@implementation INTF
- (int) foo: (int)arg1  __attribute__((deprecated)){
        return 10;
}
- (int) foo1: (int)arg1 {
        return 10;
}
- (int) foo2: (int)arg1 __attribute__((deprecated)) {
        return 10;
}
- (int) foo3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) __attribute__((ns_consumes_self)) {return 0; }
- (void) dep __attribute__((deprecated)) { } // OK private methodn
@end

@interface Foo 
- (void)doSomething1:(id)sender;
- (void)doSomething2:(id)sender;
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
- (IBAction)doSomething2:(id)sender {}
- (IBAction)doSomething3:(id)sender {}
@end

@interface NSObject @end

@interface Test : NSObject
-(id)method __attribute__((deprecated));
-(id)method1;
-(id)method2 __attribute__((aligned(16)));
- (id) method3: (int)arg1 __attribute__((aligned(16)))  __attribute__((deprecated)) __attribute__((unavailable));
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
- (id) method3: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable)) {
        return self;
}
- (id) method4: (int)arg1 __attribute__((aligned(16))) __attribute__((deprecated)) __attribute__((unavailable)) {
  return self;
}
@end

__attribute__((cdecl))  // expected-warning {{'cdecl' attribute only applies to functions and methods}}
@interface Complain 
@end

@interface rdar15450637 : NSObject
@property int p __attribute__((section("__TEXT,foo")));

- (id) IMethod :(int) count, ...  __attribute__((section("__TEXT,foo")));

+ (void) CMethod : (id) Obj __attribute__((section("__TEXT,fee")));
@end

// Section type conflicts between methods/properties and global variables
const int global1 __attribute__((section("seg1,sec1"))) = 10; // expected-note {{declared here}} expected-note {{declared here}} expected-note {{declared here}}
int global2 __attribute__((section("seg2,sec2"))) = 10;       // expected-note {{declared here}} expected-note {{declared here}} expected-note {{declared here}}

@interface section_conflicts : NSObject
@property int p1 __attribute__((section("seg1,sec1"))); // expected-error {{'p1' causes a section type conflict with 'global1'}}
@property int p2 __attribute__((section("seg2,sec2"))); // expected-error {{'p2' causes a section type conflict with 'global2'}}

- (void)imethod1 __attribute__((section("seg1,sec1"))); // expected-error {{'imethod1' causes a section type conflict with 'global1'}}
- (void)imethod2 __attribute__((section("seg2,sec2"))); // expected-error {{'imethod2' causes a section type conflict with 'global2'}}

+ (void)cmethod1:(id)Obj __attribute__((section("seg1,sec1"))); // expected-error {{'cmethod1:' causes a section type conflict with 'global1'}}
+ (void)cmethod2:(id)Obj __attribute__((section("seg2,sec2"))); // expected-error {{'cmethod2:' causes a section type conflict with 'global2'}}
@end
