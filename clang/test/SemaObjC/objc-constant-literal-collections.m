// RUN: %clang_cc1  -fsyntax-only -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -verify %s

#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@class NSString;

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSArray
+ (id)arrayWithObjects:(const id[])objects count:(NSUInteger)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

static NSArray *const array = @[ @"Hello", @"There", @"How Are You", [NSNumber numberWithInt:42] ]; // expected-error {{an array literal can only be used at file scope if its contents are all also constant literals}}
static NSArray *const array1 = @[ @"Hello", @"There", @"How Are You", @42 ];
static NSDictionary *const dictionary = @{@"Hello" : @"There", @"How Are You" : [NSNumber numberWithInt:42]}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dictionary1 = @{@"Hello" : @"There", @"How Are You" : @42};
static NSDictionary *const dictionary2 = @{@2 : @"Foo"}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}

/// For now we only support raw literals in collections not references to other varibles that *could* be modified
/// so ensure this get's downgraded to a normal runtime literal and warns even at the global scope
static NSString *const someStringConstantVar = @"foo";
static NSNumber *const someNumberConstantVar = @2;
static NSArray *const containsConstantVarReferencesArray = @[ someStringConstantVar, someNumberConstantVar ];          // expected-error {{an array literal can only be used at file scope if its contents are all also constant literals}}
static NSDictionary *const containsConstantVarReferencesDictionary = @{someStringConstantVar : someNumberConstantVar}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const containsConstantVarReferencesDictionary2 = @{@"Foo" : someNumberConstantVar};               // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}

int fooNum() {
  return 2;
}

NSNumber *foo = @(2 + fooNum()); // expected-error{{a boxed expression literal can only be used at file scope if constant}}
