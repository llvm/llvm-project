// RUN: %clang_cc1 -verify -Wswitch -triple x86_64-apple-macosx10.12 %s

enum SwitchOne {
  Unavail __attribute__((availability(macos, unavailable))),
};

void testSwitchOne(enum SwitchOne so) {
  switch (so) {} // no warning
}

enum SwitchTwo {
  Ed __attribute__((availability(macos, deprecated=10.12))),
  Vim __attribute__((availability(macos, deprecated=10.13))),
  Emacs,
};

void testSwitchTwo(enum SwitchTwo st) {
  switch (st) {} // expected-warning{{enumeration values 'Vim' and 'Emacs' not handled in switch}} expected-note{{add missing switch cases}}
}

enum SwitchThree {
  New __attribute__((availability(macos, introduced=1000))),
};

void testSwitchThree(enum SwitchThree st) {
  switch (st) {} // expected-warning{{enumeration value 'New' not handled in switch}} expected-note{{add missing switch cases}}
}
