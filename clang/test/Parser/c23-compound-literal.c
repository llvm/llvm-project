// RUN: %clang_cc1 -std=c17 -fsyntax-only -verify=c17 %s
// RUN: %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify=cxx %s

void t1(void) {
  (void)(static int){1}; // c17-error {{expected expression}} \
                         // cxx-error {{type name does not allow storage class to be specified}}
}

void t2(void) {
  (void)(register int){1}; // c17-error {{expected expression}} \
                           // cxx-error {{type name does not allow storage class to be specified}}
}

void t3(void) {
  (void)(thread_local int){1}; // c17-error {{use of undeclared identifier 'thread_local'}} \
                               // cxx-error {{type name does not allow storage class to be specified}}
}

void t4(void) {
  (void)(_Thread_local int){1}; // c17-error {{expected expression}} \
                                // cxx-error {{type name does not allow storage class to be specified}}
}

void t5(void) {
  (void)(__thread int){1}; // c17-error {{expected expression}} \
                           // cxx-error {{type name does not allow storage class to be specified}}
}

void t6(void) {
  (void)(constexpr int){1}; // c17-error {{use of undeclared identifier 'constexpr'}} \
                            // cxx-error {{type name does not allow constexpr specifier to be specified}}
}
