// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config unroll-loops=true -verify %s

void test_escaping_on_var_before_switch_case_no_crash(int c) {
  switch (c) {
    int i; // expected error{{Reached root without finding the declaration of VD}}
    case 0: {
      for (i = 0; i < 16; i++) {}
      break;
    }
  }
}
