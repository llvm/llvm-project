// RUN: %clang_analyze_cc1 -fblocks \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=osx.cocoa.MissingSuperCall \
// RUN:   -analyzer-checker=osx.cocoa.NSError \
// RUN:   -analyzer-checker=osx.ObjCProperty \
// RUN:   -analyzer-checker=osx.cocoa.RetainCount \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.core.CastToStruct \
// RUN:   -Wno-unused-value -Wno-objc-root-class -verify %s

#define SUPPRESS __attribute__((suppress))
#define SUPPRESS_SPECIFIC(...) __attribute__((suppress(__VA_ARGS__)))

@protocol NSObject
- (id)retain;
- (oneway void)release;
@end
@interface NSObject <NSObject> {
}
- (id)init;
+ (id)alloc;
@end
typedef int NSInteger;
typedef char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@class NSDictionary;
@interface NSError : NSObject <NSCopying, NSCoding> {
}
+ (id)errorWithDomain:(NSString *)domain code:(NSInteger)code userInfo:(NSDictionary *)dict;
@end

@interface NSMutableString : NSObject
@end

typedef __typeof__(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

void dereference_1() {
  int *x = 0;
  *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

void dereference_suppression_1() {
  int *x = 0;
  SUPPRESS { *x; } // no-warning
}

void dereference_2() {
  int *x = 0;
  if (*x) { // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void dereference_suppression_2() {
  int *x = 0;
  SUPPRESS if (*x) { // no-warning
  }
}

void dereference_suppression_2a() {
  int *x = 0;
  // FIXME: Implement suppressing individual checkers.
  SUPPRESS_SPECIFIC("core.NullDereference") if (*x) { // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void dereference_suppression_2b() {
  int *x = 0;
  // This is not a MallocChecker issue so it shouldn't be suppressed. (Though the attribute
  // doesn't really understand any of those arguments yet.)
  SUPPRESS_SPECIFIC("unix.Malloc") if (*x) { // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void dereference_3(int cond) {
  int *x = 0;
  if (cond) {
    (*x)++; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void dereference_suppression_3(int cond) {
  int *x = 0;
  SUPPRESS if (cond) {
    (*x)++; // no-warning
  }
}

void dereference_4() {
  int *x = 0;
  int y = *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

void dereference_suppression_4() {
  int *x = 0;
  SUPPRESS int y = *x; // no-warning
}

void dereference_5() {
  int *x = 0;
  int y = *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  int z = *x; // no-warning (duplicate)
}

void dereference_suppression_5_1() {
  int *x = 0;
  SUPPRESS int y = *x; // no-warning
  int z = *x;          // no-warning (duplicate)
}

void dereference_suppression_5_2() {
  int *x = 0;
  int y = *x;          // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  SUPPRESS int z = *x; // no-warning
}

void do_deref(int *y) {
  *y = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'y')}}
}

void dereference_interprocedural() {
  int *x = 0;
  do_deref(x);
}

void do_deref_suppressed(int *y) {
  SUPPRESS *y = 1; // no-warning
}

void dereference_interprocedural_suppressed() {
  int *x = 0;
  do_deref_suppressed(x);
}

int malloc_leak_1() {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
  return *x; // expected-warning{{Potential leak of memory pointed to by 'x'}}
}

int malloc_leak_suppression_1_1() {
  SUPPRESS int *x = (int *)malloc(sizeof(int));
  *x = 42;
  return *x;
}

int malloc_leak_suppression_1_2() {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
  SUPPRESS return *x;
}

void malloc_leak_2() {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
} // expected-warning{{Potential leak of memory pointed to by 'x'}}

void malloc_leak_suppression_2_1() {
  SUPPRESS int *x = (int *)malloc(sizeof(int));
  *x = 42;
}

void malloc_leak_suppression_2_2() SUPPRESS {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
} // no-warning

SUPPRESS void malloc_leak_suppression_2_3() {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
} // no-warning

void malloc_leak_suppression_2_4(int cond) {
  int *x = (int *)malloc(sizeof(int));
  *x = 42;
  SUPPRESS;
  // FIXME: The warning should be suppressed but dead symbol elimination
  // happens too late.
} // expected-warning{{Potential leak of memory pointed to by 'x'}}

void retain_release_leak_1() {
  [[NSMutableString alloc] init]; // expected-warning{{Potential leak of an object of type 'NSMutableString *'}}
}

void retain_release_leak_suppression_1() {
  SUPPRESS { [[NSMutableString alloc] init]; }
}

void retain_release_leak_2(int cond) {
  id obj = [[NSMutableString alloc] init]; // expected-warning{{Potential leak of an object stored into 'obj'}}
  if (cond) {
    [obj release];
  }
}

void retain_release_leak__suppression_2(int cond) {
  SUPPRESS id obj = [[NSMutableString alloc] init];
  if (cond) {
    [obj release];
  }
}

@interface UIResponder : NSObject {
}
- (char)resignFirstResponder;
@end

@interface Test : UIResponder {
}
@property(copy) NSMutableString *mutableStr;
// expected-warning@-1 {{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@end
@implementation Test

- (BOOL)resignFirstResponder {
  return 0;
} // expected-warning {{The 'resignFirstResponder' instance method in UIResponder subclass 'Test' is missing a [super resignFirstResponder] call}}

- (void)methodWhichMayFail:(NSError **)error {
  // expected-warning@-1 {{Method accepting NSError** should have a non-void return value to indicate whether or not an error occurred}}
}
@end

@interface TestSuppress : UIResponder {
}
@property(copy) SUPPRESS NSMutableString *mutableStr; // no-warning
@end
@implementation TestSuppress

- (BOOL)resignFirstResponder SUPPRESS { // no-warning
  return 0;
}

- (void)methodWhichMayFail:(NSError **)error SUPPRESS { // no-warning
}
@end

struct AB {
  int A, B;
};

struct ABC {
  int A, B, C;
};

void ast_checker_1() {
  struct AB Ab;
  struct ABC *Abc;
  Abc = (struct ABC *)&Ab; // expected-warning {{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}
}

void ast_checker_suppress_1() {
  struct AB Ab;
  struct ABC *Abc;
  SUPPRESS { Abc = (struct ABC *)&Ab; }
}

SUPPRESS int suppressed_function() {
  int *x = 0;
  return *x; // no-warning
}

SUPPRESS int suppressed_function_forward();
int suppressed_function_forward() {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

int suppressed_function_backward();
SUPPRESS int suppressed_function_backward() {
  int *x = 0;
  return *x; // no-warning
}

SUPPRESS
@interface SuppressedInterface
-(int)suppressedMethod;
-(int)regularMethod SUPPRESS;
@end

@implementation SuppressedInterface
-(int)suppressedMethod SUPPRESS {
  int *x = 0;
  return *x; // no-warning
}

// This one is NOT suppressed by the attribute on the forward declaration,
// and it's also NOT suppressed by the attribute on the entire interface.
-(int)regularMethod {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}
@end
