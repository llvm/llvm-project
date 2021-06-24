// RUN: %clang_cc1 %s -fsyntax-only -verify
__attribute__((no_profile_instrument_function))
void no_profile0(void);
#if !__has_attribute(no_profile_instrument_function)
#error "Where did the no_profile_instrument_function function attribute go?"
#endif

void no_profile1(__attribute__((no_profile_instrument_function)) int param); // expected-warning {{'no_profile_instrument_function' attribute only applies to functions}}
__attribute__((no_profile_instrument_function(""))) // expected-error {{'no_profile_instrument_function' attribute takes no arguments}}
void no_profile2(void);
void no_profile3(void) {
  __attribute__((no_profile_instrument_function)); // expected-error {{'no_profile_instrument_function' attribute cannot be applied to a statement}}
}
