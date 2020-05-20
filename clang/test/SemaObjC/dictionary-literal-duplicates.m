// RUN: %clang_cc1 -Wno-objc-root-class %s -verify
// RUN: %clang_cc1 -xobjective-c++ -Wno-objc-root-class %s -verify

#define YES __objc_yes
#define NO __objc_no

@interface NSNumber
+(instancetype)numberWithChar:(char)value;
+(instancetype)numberWithInt:(int)value;
+(instancetype)numberWithDouble:(double)value;
+(instancetype)numberWithBool:(unsigned char)value;
+(instancetype)numberWithUnsignedLong:(unsigned long)value;
+(instancetype)numberWithLongLong:(long long) value;
+(instancetype)numberWithUnsignedInt:(unsigned)value;
@end

@interface NSString
@end

@interface NSDictionary
+ (instancetype)dictionaryWithObjects:(const id[])objects
                              forKeys:(const id[])keys
                                count:(unsigned long)cnt;
@end

void test() {
  NSDictionary *t1 = @{
    @"foo" : @0, // expected-note 2 {{previous equal key is here}}
    @"foo" : @0, // expected-warning{{duplicate key in dictionary literal}}
    @("foo") : @0, // expected-warning{{duplicate key in dictionary literal}}
    @"foo\0" : @0,

    @1 : @0, // expected-note + {{previous equal key is here}}
    @YES : @0, // expected-warning{{duplicate key in dictionary literal}}
    @'\1' : @0, // expected-warning{{duplicate key in dictionary literal}}
    @1 : @0, // expected-warning{{duplicate key in dictionary literal}}
    @1ul : @0, // expected-warning{{duplicate key in dictionary literal}}
    @1ll : @0, // expected-warning{{duplicate key in dictionary literal}}
#ifdef __cplusplus
    @true : @0, // expected-warning{{duplicate key in dictionary literal}}
#endif
    @1.0 : @0, // FIXME: should warn

    @-1 : @0, // expected-note + {{previous equal key is here}}
    @4294967295u : @0, // no warning
    @-1ll : @0, // expected-warning{{duplicate key in dictionary literal}}
    @(NO-YES) : @0, // expected-warning{{duplicate key in dictionary literal}}
  };
}

#ifdef __cplusplus
template <class... Ts> void variadic(Ts... ts) {
  NSDictionary *nd = @{
    ts : @0 ...,
    @0 : ts ... // expected-warning 2 {{duplicate key in dictionary literal}} expected-note 2 {{previous equal key is here}}
  };
}

void call_variadic() {
  variadic(@0, @1, @2); // expected-note {{in instantiation}}
}
#endif
