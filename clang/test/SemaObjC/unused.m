// RUN: %clang_cc1 -verify -Wused-but-marked-unused -Wno-objc-protocol-method-implementation -Wunused -Wunused-parameter -fsyntax-only -Wno-objc-root-class %s

int printf(const char *, ...);

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello { printf("Hello, World!\n"); }
@end

int test1(void) {
  [Greeter hello];
  return 0;
}

@interface NSObject @end
@interface NSString : NSObject 
- (int)length;
@end

void test2() {
  @"pointless example call for test purposes".length; // expected-warning {{property access result unused - getters should not be used for side effects}}
}

@interface foo
- (int)meth: (int)x : (int)y : (int)z ;
@end

@implementation foo
- (int) meth: (int)x: // expected-warning {{'x' used as the name of the previous parameter rather than as part of the selector}} \
                      // expected-note {{introduce a parameter name to make 'x' part of the selector}} \
                      // expected-note {{or insert whitespace before ':' to use 'x' as parameter name and have an empty entry in the selector}}

(int)y:  // expected-warning {{'y' used as the name of the previous parameter rather than as part of the selector}} \
         // expected-note {{introduce a parameter name to make 'y' part of the selector}} \
         // expected-note {{or insert whitespace before ':' to use 'y' as parameter name and have an empty entry in the selector}}
(int) __attribute__((unused))z { return x; }
@end

//===------------------------------------------------------------------------===
// The next test shows how clang accepted attribute((unused)) on ObjC
// instance variables, which GCC does not.
//===------------------------------------------------------------------------===

#if __has_feature(attribute_objc_ivar_unused)
#define UNUSED_IVAR __attribute__((unused))
#else
#error __attribute__((unused)) not supported on ivars
#endif

@interface TestUnusedIvar {
  id y __attribute__((unused)); // no-warning
  id x UNUSED_IVAR; // no-warning
}
@end

// rdar://10777111
static NSString *x = @"hi"; // expected-warning {{unused variable 'x'}}

// rdar://12233989
@interface TestTransitiveUnused
- (void) a __attribute__((unused));
- (void) b __attribute__((unused));
@end

@interface TestTransitiveUnused(CAT)
@end

@implementation TestTransitiveUnused(CAT)
- (void) b {}
- (void) a { [self b]; }
@end

// Test that objc_precise_lifetime suppresses
// unused variable warnings.
extern void rdar15596883_foo(void);
void rdar15596883(id x) {
  __attribute__((objc_precise_lifetime)) id y = x; // no-warning
  rdar15596883_foo();
}

@interface PropertyObject : NSObject 
@property int length;
@end

@protocol P
@property int property;
@end

void test3(PropertyObject *o)
{
  [o length]; // No warning. property name used in direct method call.
}

void test4(id o)
{
  [o length]; // No warning.
}

void test5(id <P> p)
{
    [p property]; // No warning. property name used in direct method call.
}

// rdar://19773512
@interface Model
@property (nonatomic, retain, setter=setOrCreateGroup:, getter=getOrCreateGroup) id group;
@end

@implementation Model {
    id _group;
}
- (void)method {
    [self getOrCreateGroup];
    self.getOrCreateGroup; // expected-warning {{property access result unused - getters should not be used for side effects}}
    self.group; // expected-warning {{property access result unused - getters should not be used for side effects}}
    self.group = (void*)0;
    [self setOrCreateGroup : ((void*)0)];
    
}
- (id)getOrCreateGroup {
    if (!_group) {
        _group = @"group";
    }
    return _group;
}
@end

