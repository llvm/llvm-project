// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios15.1.0-simulator -fobjc-runtime=ios-15.1.0 -fobjc-arc -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -verify %s

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

// ---- Accepted: NSString literal keys with constant literal values ---------

static NSDictionary *const dASCII = @{@"a" : @1, @"m" : @2, @"z" : @3};
static NSDictionary *const dEmpty = @{};

// ---- Rejected: non-NSString-literal key -----------------------------------

static NSDictionary *const dNumberKey = @{@5 : @1}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dBoxedKey = @{@(1 + 2) : @1}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
static NSDictionary *const dBoolKey = @{@__objc_yes : @1}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}

// The error points at the specific offending key, not the whole `@{...}`.
static NSDictionary *const dSecondKeyBad = @{@"ok" : @1, @2 : @2}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}

// ---- Rejected: non-literal key (constant variable reference) --------------
//
// `someStringConstantVar` is an NSString, but it's a DeclRefExpr, not an
// ObjCStringLiteral. Hits the same "not a string literal" branch as `@5`.

static NSString *const someStringConstantVar = @"foo";
static NSDictionary *const dConstVarKey = @{someStringConstantVar : @1}; // expected-error {{a dictionary literal can only be used at file scope if its contents are all also constant literals and its keys are string literals}}
