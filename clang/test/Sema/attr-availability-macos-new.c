// RUN: %clang_cc1 "-triple" "x86_64-apple-macos10.15" -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-macos11" -DNEW -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-darwin20" -DNEW -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-macos10.15" -fsyntax-only -verify -fapplication-extension -DAPP_EXT %s

__attribute__((availability(macos,strict,introduced=10.16)))
void fNew1();
#ifndef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(macosx,strict,introduced=10.16)))
void fNew();

__attribute__((availability(macos,strict,introduced=11)))
void fNew() { }
#ifndef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(macosx,strict,deprecated=10.16)))
void fDep();

__attribute__((availability(macos,strict,deprecated=11)))
void fDep() { }
#ifdef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(macosx,strict,obsoleted=10.16)))
void fObs();

__attribute__((availability(macos,strict,obsoleted=11)))
void fObs() { }
#ifdef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(macosx_app_extension,strict,introduced=10.16)))
void fAppExt();

__attribute__((availability(macos_app_extension,strict,introduced=11)))
void fAppExt() { }
#ifdef APP_EXT
// expected-note@-2 {{here}}
#endif

void testVersionRemapping() {
  fNew1();
#ifndef NEW
  // expected-error@-2 {{'fNew1' is unavailable: introduced in macOS 11.0}}
#endif
  fNew();
#ifndef NEW
  // expected-error@-2 {{'fNew' is unavailable: introduced in macOS 11}}
#endif
  fDep();
#ifdef NEW
  // expected-warning@-2 {{'fDep' is deprecated: first deprecated in macOS 11}}
#endif
  fObs();
#ifdef NEW
  // expected-error@-2 {{'fObs' is unavailable: obsoleted in macOS 11}}
#endif

  fAppExt();
#ifdef APP_EXT
  // expected-error@-2 {{'fAppExt' is unavailable: introduced in macOS (App Extension) 11}}
#endif
}

__attribute__((availability(macosx,strict,introduced=10.16.1))) // expected-note {{here}}
void fMatchErr();

__attribute__((availability(macos,strict,introduced=11))) // expected-warning {{availability does not match previous declaration}}
void fMatchErr() { }

__attribute__((availability(macosx_app_extension,strict,introduced=10.16))) // expected-note {{here}}
void fAppExtErr();

__attribute__((availability(macos_app_extension,strict,introduced=11.1))) // expected-warning {{availability does not match previous declaration}}
void fAppExtErr() { }

__attribute__((availability(macos,introduced=11)))
void fNew2();
#ifndef NEW
  // expected-note@-2 {{'fNew2' has been marked as being introduced in macOS 11 here, but the deployment target is macOS 10.15.0}}
#endif
__attribute__((availability(macos,introduced=10.16)))
void fNew3();

__attribute__((availability(macos,introduced=12)))
void evenNewer();
#ifdef NEW
  // expected-note@-2 {{'evenNewer' has been marked as being introduced in macOS 12 here, but the deployment target is macOS 11.0.0}}
#endif

void testAvailabilityCheck() {
  if (__builtin_available(macOS 10.16, *)) {
    fNew2();
    fNew3();
  }
  if (__builtin_available(macOS 11, *)) {
    fNew2();
    fNew3();
  }
  fNew2();
#ifndef NEW
  // expected-warning@-2 {{'fNew2' is only available on macOS 11 or newer}} expected-note@-2 {{enclose}}
#endif
#ifdef NEW
  evenNewer(); // expected-warning {{'evenNewer' is only available on macOS 12 or newer}} expected-note {{enclose}}
#endif
}


