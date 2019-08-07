// Test the blacklist functionality of ASan

// RUN: echo "fun:*brokenFunction*" > %tmp
// RUN: echo "global:*badGlobal*" >> %tmp
// RUN: echo "src:*blacklist-extra.cpp" >> %tmp
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -O0 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cpp && %run %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -O1 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cpp && %run %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -O2 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cpp && %run %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -O3 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cpp && %run %t 2>&1

// badGlobal is accessed improperly, but we blacklisted it. Align
// it to make sure memory past the end of badGlobal will be in
// the same page.
__attribute__((aligned(16))) int badGlobal;
int readBadGlobal() {
  return (&badGlobal)[1];
}

// A function which is broken, but excluded in the blacklist.
int brokenFunction(int argc) {
  char x[10] = {0};
  return x[argc * 10];  // BOOM
}

// This function is defined in Helpers/blacklist-extra.cpp, a source file which
// is blacklisted by name
int externalBrokenFunction(int x);

int main(int argc, char **argv) {
  brokenFunction(argc);
  int x = readBadGlobal();
  externalBrokenFunction(argc);
  return 0;
}
