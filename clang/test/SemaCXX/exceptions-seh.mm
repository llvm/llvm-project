// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fsyntax-only -fexceptions -fobjc-exceptions -verify %s

void might_crash();

void use_seh_after_objc() {
  @try { // expected-note {{conflicting '@try' here}}
    might_crash();
  } @finally {
  }
  __try { // expected-error {{cannot use Objective-C '@try' in the same function as SEH '__try'}}
    might_crash();
  } __except(1) {
  }
}

void use_objc_after_seh() {
  __try { // expected-note {{conflicting '__try' here}}
    might_crash();
  } __except(1) {
  }
  @try { // expected-error {{cannot use Objective-C '@try' in the same function as SEH '__try'}}
    might_crash();
  } @finally {
  }
}
