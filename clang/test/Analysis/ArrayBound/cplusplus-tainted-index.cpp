// RUN: %clang_analyze_cc1 -std=c++11 -Wno-array-bounds -analyzer-config unroll-loops=true -analyzer-config security.ArrayBound:AggressiveReport=true -analyzer-checker=unix,core,security.ArrayBound  -verify %s

// Test the interactions of `security.ArrayBound` with C++ features.

void test_tainted_index_local() {
  int arr[10];
  unsigned index = 10;
  arr[index] = 7;
  // expected-warning@-1{{Out of bound access to memory after the end of 'arr'}}
}

void test_tainted_index_local_range() {
  int arr[10];
  for (unsigned index = 0; index < 11; index++)
    arr[index] = index;
    // expected-warning@-1{{Out of bound access to memory after the end of 'arr'}}
}

void test_tainted_index1(unsigned index) {
  int arr[10];
  if (index < 12)
    arr[index] = index;
  // expected-warning@-1{{Potential out of bound access to 'arr' with tainted offset}}
  if (index == 12)
    arr[index] = index;
  // expected-warning@-1{{Out of bound access to memory after the end of 'arr'}}
}

void test_tainted_index2(unsigned index) {
  int arr[10];
  if (index < 12)
    arr[index] = index;
  // expected-warning@-1{{Potential out of bound access to 'arr' with tainted offset}}
}

unsigned strlen(const char *s);
void strcpy(char *dst, char *src);
void test_ex(int argc, char *argv[]) {
  char proc_name[16];
  unsigned offset = strlen(argv[0]);
  if (offset == 16) {
    strcpy(proc_name, argv[0]);
    proc_name[offset] = 'x';
    // expected-warning@-1{{Out of bound access to memory after the end of 'proc_name'}}
  }
  if (offset <= 16) {
    strcpy(proc_name, argv[0]);
    proc_name[offset] = argv[0][16];
    // expected-warning@-1{{Potential out of bound access to the region with tainted offset}}
  }
}
