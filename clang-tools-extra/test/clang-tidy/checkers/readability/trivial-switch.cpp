// RUN: %check_clang_tidy -std=c++98-or-later %s readability-trivial-switch %t
// RUN: %check_clang_tidy -std=c++98-or-later -check-suffixes=,MACROS %s \
// RUN:   readability-trivial-switch %t -- \
// RUN:   -config="{CheckOptions: {readability-trivial-switch.IgnoreMacros: false}}"

void bad(int I) {
  switch (I) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch statement without labels has no effect [readability-trivial-switch]
  }

  switch (I) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch with default label only [readability-trivial-switch]
  default:
    break;
  }

  switch (I) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch with only one case; use an if statement [readability-trivial-switch]
  case 0:
    break;
  }

  switch (I) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch could be better written as an if-else statement [readability-trivial-switch]
  case 0:
    break;
  default:
    break;
  }
}

void good(int I) {
  switch (I) {
  case 0:
  case 1:
    break;
  }

  switch (I) {
  case 0:
    break;
  case 1:
    break;
  default:
    break;
  }
}

#define SINGLE_CASE_SWITCH(I)                                                  \
  switch (I) {                                                                 \
  case 0:                                                                      \
    break;                                                                     \
  }

void macro(int I) {
  SINGLE_CASE_SWITCH(I)
  // CHECK-MESSAGES-MACROS: :[[@LINE-1]]:3: warning: switch with only one case; use an if statement
}
