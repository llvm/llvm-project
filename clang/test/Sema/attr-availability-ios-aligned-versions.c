/// This test verifies several different patterns of iOS, and app extension, availability declarations & usages.
// RUN: %clang_cc1 "-triple" "arm64-apple-ios26" -DNEW -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "arm64-apple-ios18" -fsyntax-only -verify -fapplication-extension -DAPP_EXT %s
// RUN: %clang_cc1 "-triple" "arm64-apple-ios18" -fsyntax-only -verify %s

__attribute__((availability(ios,strict,introduced=19)))
void fNew1();
#ifndef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(ios,strict,introduced=19)))
void fNew();

__attribute__((availability(ios,strict,introduced=26)))
void fNew() { }
#ifndef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(ios,strict,deprecated=19)))
void fDep();

__attribute__((availability(ios,strict,deprecated=26)))
void fDep() { }
#ifdef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(ios,strict,obsoleted=19)))
void fObs();

__attribute__((availability(ios,strict,obsoleted=26)))
void fObs() { }
#ifdef NEW
// expected-note@-2 {{here}}
#endif

__attribute__((availability(ios_app_extension,strict,introduced=19)))
void fAppExt();

__attribute__((availability(ios_app_extension,strict,introduced=26)))
void fAppExt() { }
#ifdef APP_EXT
// expected-note@-2 {{here}}
#endif

void testVersionRemapping() {
  fNew1();
#ifndef NEW
  // expected-error@-2 {{'fNew1' is unavailable: introduced in iOS 26.0}}
#endif
  fNew();
#ifndef NEW
  // expected-error@-2 {{'fNew' is unavailable: introduced in iOS 26}}
#endif
  fDep();
#ifdef NEW
  // expected-warning@-2 {{'fDep' is deprecated: first deprecated in iOS 26}}
#endif
  fObs();
#ifdef NEW
  // expected-error@-2 {{'fObs' is unavailable: obsoleted in iOS 26}}
#endif

  fAppExt();
#ifdef APP_EXT
  // expected-error@-2 {{'fAppExt' is unavailable: introduced in iOS (App Extension) 26}}
#endif
}

__attribute__((availability(ios,strict,introduced=18.5))) // expected-note {{here}}
void fMatchErr();

__attribute__((availability(ios,strict,introduced=26))) // expected-warning {{availability does not match previous declaration}}
void fMatchErr() { }

__attribute__((availability(ios_app_extension,strict,introduced=19))) // expected-note {{here}}
void fAppExtErr();

__attribute__((availability(ios_app_extension,strict,introduced=26.1))) // expected-warning {{availability does not match previous declaration}}
void fAppExtErr() { }

__attribute__((availability(ios,introduced=26)))
void fNew2();
#ifndef NEW
  // expected-note@-2 {{'fNew2' has been marked as being introduced in iOS 26 here, but the deployment target is iOS 18}}
#endif
__attribute__((availability(ios,introduced=19)))
void fNew3();

__attribute__((availability(ios,introduced=27)))
void evenNewer();
#ifdef NEW
  // expected-note@-2 {{'evenNewer' has been marked as being introduced in iOS 27 here, but the deployment target is iOS 26}}
#endif

void testAvailabilityCheck() {
  if (__builtin_available(iOS 19, *)) {
    fNew2();
    fNew3();
  }
  if (__builtin_available(iOS 26, *)) {
    fNew2();
    fNew3();
  }
  fNew2();
#ifndef NEW
  // expected-warning@-2 {{'fNew2' is only available on iOS 26 or newer}} expected-note@-2 {{enclose}}
#endif
#ifdef NEW
  evenNewer(); // expected-warning {{'evenNewer' is only available on iOS 27 or newer}} expected-note {{enclose}}
#endif
}


