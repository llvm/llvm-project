// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -fno-constant-cfstrings -Wno-CFString-literal -verify %s

// With -fno-constant-cfstrings the keys are emitted as
// OBJC_CLASS_$_NSConstantString, which preserves the original bytes (no
// truncation) and compares them as-is at lookup time, so an ill-formed UTF-8
// key is still found at runtime. No warning is expected here -- contrast
// objc-constant-dictionary-invalid-utf8-key.m, which covers the truncating
// constant-CFString emission.
//
// (-Wno-CFString-literal silences the unrelated literal-encoding warning that
// also fires for @"\xff"; same as the companion test.)

#if __LP64__
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

// Needed to emit string literals with -fno-constant-cfstrings.
@interface NSString @end

@interface NSSimpleCString : NSString {
@protected
    char *bytes;
    unsigned int numBytes;
}
@end

@interface NSConstantString : NSSimpleCString
@end

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

// Ill-formed UTF-8 key (a lone 0xFF continuation byte): no warning, since the
// key is preserved as-is rather than truncated.
// expected-no-diagnostics
static NSDictionary *const bad = @{
    @"\xff" : @1,
    @"ok" : @2,
};
