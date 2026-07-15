// RUN: %clang_cc1 -fsyntax-only -verify=objc \
// RUN:   -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 \
// RUN:   -fobjc-constant-literals -fconstant-nsnumber-literals \
// RUN:   -fconstant-nsarray-literals -fconstant-nsdictionary-literals %s

// RUN: %clang_cc1 -fsyntax-only -verify=objc \
// RUN:   -fobjc-arc \
// RUN:   -triple arm64-apple-ios15.1.0-simulator -fobjc-runtime=ios-15.1.0 \
// RUN:   -fobjc-constant-literals -fconstant-nsnumber-literals \
// RUN:   -fconstant-nsarray-literals -fconstant-nsdictionary-literals %s

// RUN: %clang_cc1 -fsyntax-only -verify=objcxx \
// RUN:   -fobjc-arc -x objective-c++ \
// RUN:   -triple arm64-apple-ios15.1.0-simulator -fobjc-runtime=ios-15.1.0 \
// RUN:   -fobjc-constant-literals -fconstant-nsnumber-literals \
// RUN:   -fconstant-nsarray-literals -fconstant-nsdictionary-literals %s

// objcxx-no-diagnostics

#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@class NSString;

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithBool:(unsigned char)value;
@end

@interface NSArray
+ (id)arrayWithObjects:(const id[])objects count:(NSUInteger)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

static NSDictionary *const dASCII = @{@"a" : @1, @"m" : @2, @"z" : @3};
static NSDictionary *const dEmpty = @{};

static NSDictionary *const dNumberKey = @{@5 : @1}; // objc-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dBoxedKey = @{@(1 + 2) : @1}; // objc-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dBoolKey = @{@__objc_yes : @1}; // objc-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dSecondKeyBad = @{@"ok" : @1, @2 : @2}; // objc-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}

static NSString *const someStringConstantVar = @"foo";
static NSDictionary *const dConstVarKey = @{someStringConstantVar : @1}; // objc-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
