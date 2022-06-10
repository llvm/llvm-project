// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,alpha.security.ArrayBound -analyzer-store=region -verify %s
// XFAIL: *

// Once we better handle modeling of sizes of VLAs, we can pull this back
// into outofbound.c.

void sizeof_vla(int a) {
  if (a == 5) {
    char x[a];
    int y[sizeof(x)];
    y[4] = 4; // no-warning
    y[5] = 5; // expected-warning{{out-of-bound}}
  }
}

void sizeof_vla_2(int a) {
  if (a == 5) {
    char x[a];
    int y[sizeof(x) / sizeof(char)];
    y[4] = 4; // no-warning
    y[5] = 5; // expected-warning{{out-of-bound}}
  }
}

void sizeof_vla_3(int a) {
  if (a == 5) {
    char x[a];
    int y[sizeof(*&*&*&x)];
    y[4] = 4; // no-warning
    y[5] = 5; // expected-warning{{out-of-bound}}
  }
}
