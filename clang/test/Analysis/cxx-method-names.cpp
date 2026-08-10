// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,osx,alpha.unix,optin.taint -verify %s
// expected-no-diagnostics

class Evil {
public:
  void system(int); // taint checker
  void malloc(void *); // taint checker, malloc checker
  void free(); // malloc checker, keychain checker
  void fopen(); // stream checker
  void feof(int, int); // stream checker
  void open(); // unix api checker
};

void test(Evil &E) {
  // no warnings, no crashes
  E.system(0);
  E.malloc(0);
  E.free();
  E.fopen();
  E.feof(0,1);
  E.open();
}
