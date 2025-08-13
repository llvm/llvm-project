// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/diagnose-missing-import \
// RUN:   -Werror=implicit-function-declaration -fsyntax-only \
// RUN:   -fimplicit-module-maps -verify %s
@import NCI;

void foo(void) {
  XYZLogEvent(xyzRiskyCloseOpenParam, xyzRiskyCloseOpenParam); // expected-error {{call to undeclared function 'XYZLogEvent'; ISO C99 and later do not support implicit function declarations}} \
                                                                  expected-error {{declaration of 'XYZLogEvent' must be imported}} \
                                                                  expected-error {{declaration of 'xyzRiskyCloseOpenParam' must be imported from module 'NCI.A'}}
}

// expected-note@Inputs/diagnose-missing-import/a.h:5 {{declaration here is not visible}}
// expected-note@Inputs/diagnose-missing-import/a.h:6 {{declaration here is not visible}}

