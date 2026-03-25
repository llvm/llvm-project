// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu future \
// RUN:   -target-feature +vsx -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu future \
// RUN:   -target-feature +vsx -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr10 \
// RUN:   -target-feature +vsx -fsyntax-only -verify=pwr10 %s

vector unsigned char vuca, vucb, vucc;
int ia;

void test_xxmulmul() {
  __builtin_xxmulmul(vuca, vucb, 8);   // expected-error {{argument value 8 is outside the valid range [0, 7]}} \
                                       // pwr10-error {{'__builtin_xxmulmul' needs target feature isa-future-instructions}}
  __builtin_xxmulmul(vuca, vucb, -1);  // expected-error {{argument value -1 is outside the valid range [0, 7]}} \
                                       // pwr10-error {{'__builtin_xxmulmul' needs target feature isa-future-instructions}}
  __builtin_xxmulmul(vuca, vucb, ia);  // expected-error {{argument to '__builtin_xxmulmul' must be a constant integer}} \
                                       // pwr10-error {{'__builtin_xxmulmul' needs target feature isa-future-instructions}}
}

void test_xxmulmulhiadd() {
  __builtin_xxmulmulhiadd(vuca, vucb, 2, 0, 0);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                                 // pwr10-error {{'__builtin_xxmulmulhiadd' needs target feature isa-future-instructions}}
  __builtin_xxmulmulhiadd(vuca, vucb, 0, 2, 0);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                                 // pwr10-error {{'__builtin_xxmulmulhiadd' needs target feature isa-future-instructions}}
  __builtin_xxmulmulhiadd(vuca, vucb, 0, 0, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                                 // pwr10-error {{'__builtin_xxmulmulhiadd' needs target feature isa-future-instructions}}
  __builtin_xxmulmulhiadd(vuca, vucb, ia, 0, 0); // expected-error {{argument to '__builtin_xxmulmulhiadd' must be a constant integer}} \
                                                 // pwr10-error {{'__builtin_xxmulmulhiadd' needs target feature isa-future-instructions}}
}

void test_xxmulmulloadd() {
  __builtin_xxmulmulloadd(vuca, vucb, 2, 0);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                              // pwr10-error {{'__builtin_xxmulmulloadd' needs target feature isa-future-instructions}}
  __builtin_xxmulmulloadd(vuca, vucb, 0, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                              // pwr10-error {{'__builtin_xxmulmulloadd' needs target feature isa-future-instructions}}
  __builtin_xxmulmulloadd(vuca, vucb, ia, 0); // expected-error {{argument to '__builtin_xxmulmulloadd' must be a constant integer}} \
                                              // pwr10-error {{'__builtin_xxmulmulloadd' needs target feature isa-future-instructions}}
}

void test_xxssumudm() {
  __builtin_xxssumudm(vuca, vucb, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                       // pwr10-error {{'__builtin_xxssumudm' needs target feature isa-future-instructions}}
  __builtin_xxssumudm(vuca, vucb, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}} \
                                       // pwr10-error {{'__builtin_xxssumudm' needs target feature isa-future-instructions}}
  __builtin_xxssumudm(vuca, vucb, ia); // expected-error {{argument to '__builtin_xxssumudm' must be a constant integer}} \
                                       // pwr10-error {{'__builtin_xxssumudm' needs target feature isa-future-instructions}}
}

void test_xxssumudmc() {
  __builtin_xxssumudmc(vuca, vucb, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                        // pwr10-error {{'__builtin_xxssumudmc' needs target feature isa-future-instructions}}
  __builtin_xxssumudmc(vuca, vucb, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}} \
                                        // pwr10-error {{'__builtin_xxssumudmc' needs target feature isa-future-instructions}}
  __builtin_xxssumudmc(vuca, vucb, ia); // expected-error {{argument to '__builtin_xxssumudmc' must be a constant integer}} \
                                        // pwr10-error {{'__builtin_xxssumudmc' needs target feature isa-future-instructions}}
}

void test_xxssumudmcext() {
  __builtin_xxssumudmcext(vuca, vucb, vucc, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}} \
                                                 // pwr10-error {{'__builtin_xxssumudmcext' needs target feature isa-future-instructions}}
  __builtin_xxssumudmcext(vuca, vucb, vucc, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}} \
                                                 // pwr10-error {{'__builtin_xxssumudmcext' needs target feature isa-future-instructions}}
  __builtin_xxssumudmcext(vuca, vucb, vucc, ia); // expected-error {{argument to '__builtin_xxssumudmcext' must be a constant integer}} \
                                                 // pwr10-error {{'__builtin_xxssumudmcext' needs target feature isa-future-instructions}}
}
