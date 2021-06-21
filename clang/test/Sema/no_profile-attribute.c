// RUN: %clang_cc1 %s -fsyntax-only -verify
__attribute__((no_profile))
void no_profile0(void);
#if !__has_attribute(no_profile)
#error "Where did the no_profile function attribute go?"
#endif

void no_profile1(__attribute__((no_profile)) int param); // expected-warning {{'no_profile' attribute only applies to functions}}
__attribute__((no_profile(""))) // expected-error {{'no_profile' attribute takes no arguments}}
void no_profile2(void);
void no_profile3(void) {
  __attribute__((no_profile)); // expected-error {{'no_profile' attribute cannot be applied to a statement}}
}
