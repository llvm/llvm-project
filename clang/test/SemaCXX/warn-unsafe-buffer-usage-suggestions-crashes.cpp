// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            %s -verify %s

char * unsafe_pointer; // expected-warning{{'unsafe_pointer' is an unsafe pointer used for buffer access}}

void test(char * param) {
}

void dre_parenthesized() {
  test(&(unsafe_pointer)[1]); // no-crash // expected-note{{used in buffer access here}}
}
