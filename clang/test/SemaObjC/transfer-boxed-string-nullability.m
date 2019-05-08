// RUN: %clang_cc1 -fblocks -fobjc-arc -Wnullable-to-nonnull-conversion -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wnullable-to-nonnull-conversion -fsyntax-only -verify -Wno-objc-root-class -DNOWARN %s

@interface NSString

+ (NSString*
#ifndef NOWARN
  _Nullable
#else
  _Nonnull
#endif
) stringWithUTF8String:(const char*)x;

@end

void takesNonNull(NSString * _Nonnull ptr);

void testBoxedString() {
  // No diagnostic emitted as this doesn't need a stringWithUTF8String message
  // send.
  takesNonNull(@("hey"));
  takesNonNull(@(u8"hey"));

  // If the string isn't a valid UTF-8 string, a diagnostic is emitted since the
  // boxed expression turns into a message send.
  takesNonNull(@(u8"\xFF")); // expected-warning {{string is ill-formed as UTF-8}}
  takesNonNull(@(u8"\xC0\x80")); // expected-warning {{string is ill-formed as UTF-8}}

  const char *str = "hey";
  takesNonNull([NSString stringWithUTF8String:str]);
  takesNonNull(@(str));
#ifndef NOWARN
  // expected-warning@-7 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
  // expected-warning@-7 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
  // expected-warning@-5 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
  // expected-warning@-5 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
#endif
}
