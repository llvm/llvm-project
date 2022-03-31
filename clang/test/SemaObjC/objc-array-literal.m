// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://10111397
// RUN: %clang_cc1  -fsyntax-only -triple i386-apple-macosx10.9.0 -fobjc-runtime=macosx-fragile-10.9.0 -fobjc-subscripting-legacy-runtime -verify %s
// rdar://15363492

#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

void checkNSArrayUnavailableDiagnostic(void) {
  id obj;
  id arr = @[obj]; // expected-error {{definition of class NSArray must be available to use Objective-C array literals}}
}

@class NSArray; // expected-note {{forward declaration of class here}}

void checkNSArrayFDDiagnostic(void) {
  id obj;
  id arr = @[obj]; // expected-error {{definition of class NSArray must be available to use Objective-C array literals}}
}

@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

@class NSFastEnumerationState;

@protocol NSFastEnumeration

- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id [])buffer count:(NSUInteger)len;

@end

@interface NSNumber 
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSArray <NSFastEnumeration>
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end


int main(void) {
 NSArray *array = @[@"Hello", @"There", @"How Are You", [NSNumber numberWithInt:42]];

  for (id string in array)
    NSLog(@"%@\n", string);

  NSArray *array1 = @["Forgot"]; // expected-error {{string literal must be prefixed by '@' in a collection}}

  const char *blah;
  NSArray *array2 = @[blah]; // expected-error{{collection element of type 'const char *' is not an Objective-C object}}
}

// rdar://14303083
id Test14303083(void) {
  id obj = @[ @"A", (@"B" @"C")];
  return @[ @"A", @"B" @"C"]; // expected-warning {{concatenated NSString literal for an NSArray expression - possibly missing a comma}}
}
id radar15147688(void) {
#define R15147688_A @"hello"
#define R15147688_B "world"
#define CONCATSTR R15147688_A R15147688_B
  id x = @[ @"stuff", CONCATSTR ]; // no-warning
  x = @[ @"stuff", @"hello" "world"]; // expected-warning {{concatenated NSString literal for an NSArray expression}}
  return x;
}

enum XXXYYYZZZType { XXXYYYZZZTypeAny }; // expected-note {{'XXXYYYZZZTypeAny' declared here}}
void foo(void) {
  NSArray *array = @[
    @(XXXYYYZZZTypeA),                 // expected-error {{use of undeclared identifier 'XXXYYYZZZTypeA'; did you mean 'XXXYYYZZZTypeAny'}}
    @(XXXYYYZZZTypeSomethingSomething) // expected-error {{use of undeclared identifier 'XXXYYYZZZTypeSomethingSomething'}}
  ];
}
