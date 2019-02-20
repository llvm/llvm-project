// RUN: %clang_cc1 -triple arm64-apple-tvos12.0 -fsyntax-only -verify %s

void explicit() __attribute__((availability(tvos, introduced=11.0, deprecated=12.0))); // expected-note {{marked deprecated here}}
void inferred() __attribute__((availability(ios, introduced=11.0, deprecated=12.0))); // expected-note {{marked deprecated here}}
void explicitOverInferred()
__attribute__((availability(ios, introduced=11.0, deprecated=12.0)))
__attribute__((availability(tvos, introduced=11.0)));
void explicitOverInferred2()
__attribute__((availability(tvos, introduced=11.0)))
__attribute__((availability(ios, introduced=11.0, deprecated=12.0)));

void simpleUsage() {
  explicit(); // expected-warning{{'explicit' is deprecated: first deprecated in tvOS 12.0}}
  inferred(); // expected-warning{{'inferred' is deprecated: first deprecated in tvOS 12.0}}
  // ok, not deprecated for tvOS.
  explicitOverInferred();
  explicitOverInferred2();
}

#pragma clang attribute push (__attribute__((availability(tvos, introduced=11.0, deprecated=12.0))), apply_to=function)

void explicitFromPragma(); // expected-note {{marked deprecated here}}
void explicitWinsOverExplicitFromPragma() __attribute__((availability(tvos, introduced=11.0)));
void implicitLosesOverExplicitFromPragma() __attribute__((availability(ios, introduced=11.0))); // expected-note {{marked deprecated here}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((availability(ios, introduced=11.0, deprecated=12.0))), apply_to=function)

void implicitFromPragma(); // expected-note {{marked deprecated here}}
void explicitWinsOverImplicitFromPragma() __attribute__((availability(tvos, introduced=11.0)));
void implicitWinsOverImplicitFromPragma() __attribute__((availability(ios, introduced=11.0)));

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((availability(tvos, introduced=11.0, deprecated=12.0))), apply_to=function)
#pragma clang attribute push (__attribute__((availability(ios, introduced=11.0, deprecated=11.3))), apply_to=function)

void pragmaExplicitWinsOverPragmaImplicit(); // expected-note {{marked deprecated here}}

#pragma clang attribute pop
#pragma clang attribute pop

void pragmaUsage() {
  explicitFromPragma(); // expected-warning {{'explicitFromPragma' is deprecated: first deprecated in tvOS 12.0}}
  explicitWinsOverExplicitFromPragma(); // ok
  implicitLosesOverExplicitFromPragma(); // expected-warning {{'implicitLosesOverExplicitFromPragma' is deprecated: first deprecated in tvOS 12.0}}

  implicitFromPragma(); // expected-warning {{'implicitFromPragma' is deprecated: first deprecated in tvOS 12.0}}
  explicitWinsOverImplicitFromPragma(); // ok
  implicitWinsOverImplicitFromPragma(); // ok
  pragmaExplicitWinsOverPragmaImplicit(); // expected-warning {{'pragmaExplicitWinsOverPragmaImplicit' is deprecated: first deprecated in tvOS 12.0}}
}
