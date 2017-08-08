// RUN: %clang_cc1 -fsyntax-only -Wunguarded-availability -Wno-unsupported-availability-guard -triple x86_64-apple-macosx10.10.0 -verify %s

void f() {

  if (@available(macos 10.12, *)) {}
  else if (@available(macos 10.11, *)) {}
  else {}

  (void)__builtin_available(ios 8, macos 10.10, *);

  (void)@available(macos 10.11); // expected-error{{must handle potential future platforms with '*'}}
  (void)@available(macos 10.11, macos 10.11, *); // expected-error{{version for 'macos' already specified}}

  (void)@available(erik_os 10.11, *); // expected-error{{unrecognized platform name erik_os}}

  (void)@available(erik_os 10.10, hat_os 1.0, *); // expected-error 2 {{unrecognized platform name}}

  (void)@available(); // expected-error{{expected a platform name here}}
  (void)@available(macos 10.10,); // expected-error{{expected a platform name here}}
  (void)@available(macos); // expected-error{{expected a version}}
  (void)@available; // expected-error{{expected '('}}
}

void prettyPlatformNames() {
  (void)@available(iOS 8, tvOS 10, watchOS 3, macOS 10.11, *);
  (void)__builtin_available(iOSApplicationExtension 8, tvOSApplicationExtension 10,
                   watchOSApplicationExtension 3, macOSApplicationExtension 10.11, *);
}

#if __has_builtin(__builtin_available)
#error expected
// expected-error@-1 {{expected}}
#endif
