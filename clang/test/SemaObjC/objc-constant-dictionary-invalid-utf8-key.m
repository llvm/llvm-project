// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -Wno-CFString-literal -verify %s

// A constant dictionary stores and looks up its keys as UTF-16, truncating any
// key at the first ill-formed UTF-8 byte. Warn when that happens, since the
// truncated key generally cannot be found at runtime.

#if __LP64__
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

// Ill-formed UTF-8 key (a lone 0xFF continuation byte) in a constant dictionary.
static NSDictionary *const bad = @{
    @"\xff" : @1, // expected-warning {{dictionary key is ill-formed as UTF-8 and will be truncated in a static constant dictionary, so it may not be found at runtime}}
    @"ok" : @2,
};

// Well-formed keys (including non-ASCII and astral) must NOT warn.
static NSDictionary *const good = @{
    @"ascii" : @1,
    @"café" : @2,
    @"\U0001F600" : @3,
};

// A runtime (non-constant) dictionary keeps the original NSString and is not
// truncated, so it must NOT warn even with an ill-formed key.
NSDictionary *runtime(NSNumber *n) {
  return @{
      @"\xff" : n,
      @"ok" : n,
  };
}
