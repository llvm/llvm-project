// RUN: %clang_cc1 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -std=c++20 -verify=expected %s
// RUN: %clang_cc1 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -mllvm -debug-only=SafeBuffers \
// RUN:            -std=c++20 -verify=expected,debug %s

// A generic -debug would also enable our notes. This is probably fine.
//
// RUN: %clang_cc1 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -std=c++20 -mllvm -debug \
// RUN:            -verify=expected,debug %s

// This test file checks the behavior under the assumption that no fixits
// were emitted for the test cases. If -Wunsafe-buffer-usage is improved
// to support these cases (thus failing the test), the test should be changed
// to showcase a different unsupported example.
//
// RUN: %clang_cc1 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -mllvm -debug-only=SafeBuffers \
// RUN:            -std=c++20 -fdiagnostics-parseable-fixits %s \
// RUN:            2>&1 | FileCheck %s
// CHECK-NOT: fix-it:

// This debugging facility is only available in debug builds.
//
// REQUIRES: asserts

void foo() {
  int *x = new int[10]; // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  x[5] = 10;            // expected-note{{used in buffer access here}}
  int z = x[-1];        // expected-note{{used in buffer access here}} \
                        // debug-note{{safe buffers debug: gadget 'ULCArraySubscript' refused to produce a fix}}
}

void failed_multiple_decl() {
  int *a = new int[4], b;  // expected-warning{{'a' is an unsafe pointer used for buffer access}} \
                          // debug-note{{safe buffers debug: failed to produce fixit for declaration 'a' : multiple VarDecls}}
  a[4] = 3;  // expected-note{{used in buffer access here}}
}

void failed_param_var_decl(int *a =new int[3]) {  // expected-warning{{'a' is an unsafe pointer used for buffer access}} \
  // debug-note{{safe buffers debug: failed to produce fixit for declaration 'a' : has default arg}}
  a[4] = 6;  // expected-note{{used in buffer access here}}
}

void unclaimed_use() {
  int *a = new int[3];  // expected-warning{{'a' is an unsafe pointer used for buffer access}}
  a[2] = 9;  // expected-note{{used in buffer access here}}
  int *b = a++;  // expected-note{{used in pointer arithmetic here}} \
  // debug-note{{safe buffers debug: failed to produce fixit for 'a' : has an unclaimed use}}
}

void implied_unclaimed_var(int *b) {  // expected-warning{{'b' is an unsafe pointer used for buffer access}}
  int *a = new int[3];  // expected-warning{{'a' is an unsafe pointer used for buffer access}}
  a[4] = 7;  // expected-note{{used in buffer access here}}
  a = b;  // debug-note{{safe buffers debug: gadget 'PtrToPtrAssignment' refused to produce a fix}}
  b++;  // expected-note{{used in pointer arithmetic here}} \
        // debug-note{{safe buffers debug: failed to produce fixit for 'b' : has an unclaimed use}}
}

int *a = new int[3];  // expected-warning{{'a' is an unsafe pointer used for buffer access}} \
// debug-note{{safe buffers debug: failed to produce fixit for 'a' : neither local nor a parameter}}
void test_globals() {
  a[7] = 4;  // expected-note{{used in buffer access here}}
}

void test_decomp_decl() {
  int a[2] = {1, 2};
  auto [x, y] = a;
  x = 9;
}

void test_claim_use_multiple() {
  int *a = new int[8];  // expected-warning{{'a' is an unsafe pointer used for buffer access}}
  a[6] = 9;  // expected-note{{used in buffer access here}}
  a++;  // expected-note{{used in pointer arithmetic here}} \
  // debug-note{{safe buffers debug: failed to produce fixit for 'a' : has an unclaimed use}}
}

struct S
{
    int *x;
};
 
S f() { return S{new int[4]}; }

void test_struct_claim_use() {
  auto [x] = f();
  x[6] = 8;  // expected-warning{{unsafe buffer access}}
  x++;  // expected-warning{{unsafe pointer arithmetic}}
}

void use(int*);
void array2d(int idx) {
  int buffer[10][5]; // expected-warning{{'buffer' is an unsafe buffer that does not perform bounds checks}}
  use(buffer[idx]);  // expected-note{{used in buffer access here}} \
  // debug-note{{safe buffers debug: failed to produce fixit for 'buffer' : has an unclaimed use}}
}
