// RUN: %clang_cc1 -triple armv7 -std=c++14 -x c++ %s -fsyntax-only
// expected-no-diagnostics

void deduce() {
  auto single_int = [](int i) __attribute__ (( pcs("aapcs") )) {
    return i;
  };
  auto multiple_int = [](int i) __attribute__ (( pcs("aapcs") ))
                                __attribute__ (( pcs("aapcs") )) {
    return i;
  };

  auto single_void = []() __attribute__ (( pcs("aapcs") )) { };
  auto multiple_void = []() __attribute__ (( pcs("aapcs") ))
                            __attribute__ (( pcs("aapcs") )) { };
}

auto ( __attribute__ (( pcs("aapcs") )) single_attribute() ) { }
auto ( ( __attribute__ (( pcs("aapcs") )) ( ( __attribute__ (( pcs("aapcs") )) multiple_attributes() ) ) ) ) { }

