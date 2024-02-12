// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -verify %s

// CHECK-NOT: [-Wunsafe-buffer-usage]


void foo(unsigned idx) {
  int buffer[10];         // expected-warning{{'buffer' is an unsafe buffer that does not perform bounds checks}}
                          // expected-note@-1{{change type of 'buffer' to 'std::array' to label it for hardening}}
  buffer[idx] = 0;        // expected-note{{used in buffer access here}}
}

int global_buffer[10];    // expected-warning{{'global_buffer' is an unsafe buffer that does not perform bounds checks}}
void foo2(unsigned idx) {
  global_buffer[idx] = 0;        // expected-note{{used in buffer access here}}
}

struct Foo {
  int member_buffer[10];
};
void foo2(Foo& f, unsigned idx) {
  f.member_buffer[idx] = 0; // expected-warning{{unsafe buffer access}}
}
