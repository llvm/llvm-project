// RUN: %clang_cc1 -fsyntax-only -verify -Wassign-enum -Wswitch-enum -Wcovered-switch-default -std=c++11 %s

enum Enum {
  A0 = 1, A1 = 10
};

enum __attribute__((enum_extensibility(closed))) EnumClosed {
  B0 = 1, B1 = 10
};

enum [[clang::enum_extensibility(open)]] EnumOpen {
  C0 = 1, C1 = 10
};

enum __attribute__((flag_enum)) EnumFlag {
  D0 = 1, D1 = 8
};

enum __attribute__((flag_enum,enum_extensibility(closed))) EnumFlagClosed {
  E0 = 1, E1 = 8
};

enum __attribute__((flag_enum,enum_extensibility(open))) EnumFlagOpen {
  F0 = 1, F1 = 8
};

void test() {
  enum Enum t0;

  switch (t0) { // expected-warning{{enumeration value 'A1' not handled in switch}}
  case A0: break;
  case 16: break; // expected-warning{{case value not in enumerated type}}
  }

  switch (t0) {
  case A0: break;
  case A1: break;
  default: break; // expected-warning{{default label in switch which covers all enumeration}}
  }

  enum EnumClosed t1;

  switch (t1) { // expected-warning{{enumeration value 'B1' not handled in switch}}
  case B0: break;
  case 16: break; // expected-warning{{case value not in enumerated type}}
  }

  switch (t1) {
  case B0: break;
  case B1: break;
  default: break; // expected-warning{{default label in switch which covers all enumeration}}
  }

  enum EnumOpen t2;

  switch (t2) { // expected-warning{{enumeration value 'C1' not handled in switch}}
  case C0: break;
  case 16: break;
  }

  switch (t2) {
  case C0: break;
  case C1: break;
  default: break;
  }

  enum EnumFlag t3;

  switch (t3) { // expected-warning{{enumeration value 'D1' not handled in switch}}
  case D0: break;
  case 9: break;
  case 16: break; // expected-warning{{case value not in enumerated type}}
  }

  switch (t3) {
  case D0: break;
  case D1: break;
  default: break;
  }

  enum EnumFlagClosed t4;

  switch (t4) { // expected-warning{{enumeration value 'E1' not handled in switch}}
  case E0: break;
  case 9: break;
  case 16: break; // expected-warning{{case value not in enumerated type}}
  }

  switch (t4) {
  case E0: break;
  case E1: break;
  default: break;
  }

  enum EnumFlagOpen t5;

  switch (t5) { // expected-warning{{enumeration value 'F1' not handled in switch}}
  case F0: break;
  case 9: break;
  case 16: break;
  }

  switch (t5) {
  case F0: break;
  case F1: break;
  default: break;
  }
}
