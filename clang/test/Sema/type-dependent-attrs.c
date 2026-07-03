// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

int open() { return 0; }
void close(typeof(open()) *) {}

void cleanup_attr() {
  int fd_int [[gnu::cleanup(close)]] = open();
  auto fd_auto [[gnu::cleanup(close)]] = open();
  float fd_invalid [[gnu::cleanup(close)]] = open(); // expected-error {{'cleanup' function 'close' parameter has type 'typeof (open()) *' (aka 'int *') which is incompatible with type 'float *'}}
}
