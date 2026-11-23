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

// In C++ these file-scope initializers are dynamic initializers rather than
// constant ones, so none of the diagnostics below apply.
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

int foo(void);

static NSString *const someStringConstantVar = @"foo";

// ---- Accepted: string-literal keys with constant literal values -----------
// These are constant, so they must be diagnosed under neither triple. The `id`
// case additionally wraps the literal in an implicit BitCast, which must not
// stop it from being recognized as a constant literal.

static NSDictionary *const dASCII = @{@"a" : @1, @"m" : @2, @"z" : @3};
static NSDictionary *const dEmpty = @{};
static NSArray *const aOK = @[@1, @2, @3];
static id const dOKAsId = @{@"a" : @1};

// ---- Rejected: non-string-literal key -------------------------------------
// The error must point at the offending key, not the whole `@{...}` and not the
// value. Splitting each element across lines lets -verify confirm the caret
// lands on the key. This must hold with and without ARC.

static NSDictionary *const dNumberKey = @{
    @5 // objc-error {{its keys are string literals}}
    : @1};

static NSDictionary *const dBoxedKey = @{
    @(1 + 2) // objc-error {{its keys are string literals}}
    : @1};

static NSDictionary *const dBoolKey = @{
    @__objc_yes // objc-error {{its keys are string literals}}
    : @1};

// The offending key is the second entry; the first (valid) entry is not blamed.
static NSDictionary *const dSecondKeyBad = @{
    @"ok" : @1,
    @2 // objc-error {{its keys are string literals}}
    : @2};

// A constant NSString variable is not a string *literal*, so it is rejected as
// a key and the caret points at it.
static NSDictionary *const dConstVarKey = @{
    someStringConstantVar // objc-error {{its keys are string literals}}
    : @1};

// ---- Rejected: valid string key but non-constant value --------------------
// The key is a valid string literal; the value is the culprit, so the caret
// must point at the value and the key must not be blamed.

static NSDictionary *const dValueBad = @{
    @"a" :
    @(foo())}; // objc-error {{its keys are string literals}}

// Even when an earlier entry is fully valid, the caret points at the first bad
// value rather than the whole literal or the good entry.
static NSDictionary *const dMixedValueBad = @{
    @"a" : @1,
    @"b" :
    @(foo())}; // objc-error {{its keys are string literals}}

// A value that is constant but not itself a literal (a reference to another
// constant) is likewise rejected, pointing at the value.
static NSDictionary *const dNonLiteralValue = @{
    @"a" :
    someStringConstantVar}; // objc-error {{its keys are string literals}}

// ---- Rejected: id-typed variable under ARC --------------------------------
// Declaring the variable `id` wraps the literal in an implicit BitCast. The
// diagnostic must still be the dictionary-specific one (not the generic C
// "initializer element is not a compile-time constant") and must point at the
// offending key.

static id const dNumberKeyAsId = @{
    @5 // objc-error {{its keys are string literals}}
    : @1};

// ---- Rejected: non-constant array element ---------------------------------
// The array diagnostic points at the specific offending element.

static NSArray *const aValueBad = @[
    @1,
    @(foo())]; // objc-error {{an array literal can only be used at file scope}}

// ---- Rejected: parenthesized literal --------------------------------------
// A parenthesized initializer such as `(@{...})` adds a `ParenExpr`; combined
// with the implicit BitCast for an `id`-typed variable it would otherwise hide
// the literal and fall back to the generic diagnostic. The specific-culprit
// reporting must still fire through parentheses (including nested ones) for both
// dictionary- and id-typed variables.

static NSDictionary *const dParenValueBad = (@{
    @"a" :
    @(foo())}); // objc-error {{its keys are string literals}}

static id const dParenIdValueBad = (@{
    @"a" :
    @(foo())}); // objc-error {{its keys are string literals}}

static id const dNestedParenIdValueBad = ((@{
    @"a" :
    @(foo())})); // objc-error {{its keys are string literals}}

static id const aParenIdValueBad = (@[
    @1,
    @(foo())]); // objc-error {{an array literal can only be used at file scope}}
