// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/diagnose-missing-import \
// RUN:   -Werror=implicit-function-declaration -fsyntax-only \
// RUN:   -fimplicit-module-maps -verify %s
@import NCI;

void foo() {
  XYZLogEvent(xyzRiskyCloseOpenParam, xyzRiskyCloseOpenParam); // expected-error {{implicit declaration of function 'XYZLogEvent'}} expected-error {{declaration of 'XYZLogEvent' must be imported}} expected-error {{declaration of 'xyzRiskyCloseOpenParam' must be imported from module 'NCI.A'}} expected-error {{declaration of 'xyzRiskyCloseOpenParam' must be imported from module 'NCI.A'}}
}

// expected-note@Inputs/diagnose-missing-import/a.h:5 {{previous declaration is here}}
// expected-note@Inputs/diagnose-missing-import/a.h:5 {{previous declaration is here}}
// expected-note@Inputs/diagnose-missing-import/a.h:6 {{previous declaration is here}}

