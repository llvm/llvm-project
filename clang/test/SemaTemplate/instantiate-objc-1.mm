// RUN: %clang_cc1 -fsyntax-only -verify %s

// Obj-C string literal expressions
template <typename T> struct StringTest {
  void f() {
    (void)@"Hello";
  }
};

template struct StringTest<int>;
template struct StringTest<double>;

// @selector expressions
template <typename T> struct SelectorTest {
  SEL f() {
    return @selector(multiple:arguments:);
  }
  SEL f2() {
    return @selector(multiple:arguments:);
  }
};

template struct SelectorTest<int>;
template struct SelectorTest<double>;

// @protocol expressions
@protocol P
@end

template <typename T> struct ProtocolTest {
  void f() {
    (void)@protocol(P);
  }
};

template struct ProtocolTest<int>;
template struct ProtocolTest<double>;

// @encode expressions
template <typename T> struct EncodeTest {
  static const char *encode(T t) { 
    return @encode(T);
  }
};

template struct EncodeTest<int>;
template struct EncodeTest<double>;
template struct EncodeTest<wchar_t>;

// @() boxing expressions.
template <typename T> struct BoxingTest {
  static id box(T value) {
    return @(value);                     // expected-error {{illegal type 'int *' used in a boxed expression}} \
                                         // expected-error {{illegal type 'long double' used in a boxed expression}}
  }
};

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSString
+ (id)stringWithUTF8String:(const char *)str;
@end

template struct BoxingTest<int>;
template struct BoxingTest<const char *>;
template struct BoxingTest<int *>;        // expected-note {{in instantiation of member function 'BoxingTest<int *>::box' requested here}}
template struct BoxingTest<long double>;  // expected-note {{in instantiation of member function 'BoxingTest<long double>::box' requested here}}
