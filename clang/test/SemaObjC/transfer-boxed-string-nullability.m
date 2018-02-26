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
  const char *str = "hey";
  takesNonNull([NSString stringWithUTF8String:str]);
  takesNonNull(@(str));
#ifndef NOWARN
  // expected-warning@-3 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
  // expected-warning@-3 {{implicit conversion from nullable pointer 'NSString * _Nullable' to non-nullable pointer type 'NSString * _Nonnull'}}
#else
  // expected-no-diagnostics
#endif
}
