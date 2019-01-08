// RUN: %clang_cc1 %s -verify -fsyntax-only

void good() {
  int dont_initialize_me __attribute((uninitialized));
}

void bad() {
  int im_bad __attribute((uninitialized("zero")));  // expected-error {{'uninitialized' attribute takes no arguments}}
  static int im_baaad __attribute((uninitialized)); // expected-warning {{'uninitialized' attribute only applies to local variables}}
}

extern int come_on __attribute((uninitialized));                    // expected-warning {{'uninitialized' attribute only applies to local variables}}
int you_know __attribute((uninitialized));                          // expected-warning {{'uninitialized' attribute only applies to local variables}}
static int and_the_whole_world_has_to __attribute((uninitialized)); // expected-warning {{'uninitialized' attribute only applies to local variables}}

void answer_right_now() __attribute((uninitialized)) {}                        // expected-warning {{'uninitialized' attribute only applies to local variables}}
void just_to_tell_you_once_again(__attribute((uninitialized)) int whos_bad) {} // expected-warning {{'uninitialized' attribute only applies to local variables}}

struct TheWordIsOut {
  __attribute((uninitialized)) int youre_doin_wrong; // expected-warning {{'uninitialized' attribute only applies to local variables}}
} __attribute((uninitialized));                      // expected-warning {{'uninitialized' attribute only applies to local variables}}
