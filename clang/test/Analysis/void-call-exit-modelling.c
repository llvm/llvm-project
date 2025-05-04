// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -analyzer-output text -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t size);

void inf_loop_break_callee() {
  void* data = malloc(10); // expected-note{{Memory is allocated}}
  while (1) { // expected-note{{Loop condition is true}}
    (void)data;
    break; // No note that we jump to the line above from this break
  } // expected-note@-1{{Execution jumps to the end of the function}}
} // expected-warning{{Potential leak of memory pointed to by 'data'}}
// expected-note@-1  {{Potential leak of memory pointed to by 'data'}}

void inf_loop_break_caller() {
  inf_loop_break_callee(); // expected-note{{Calling 'inf_loop_break_callee'}}
}

void inf_loop_break_top() {
  void* data = malloc(10); // expected-note{{Memory is allocated}}
  while (1) { // expected-note{{Loop condition is true}}
    (void)data;
    break; // No note that we jump to the line above from this break
  } // expected-note@-1{{Execution jumps to the end of the function}}
} // expected-warning{{Potential leak of memory pointed to by 'data'}}
// expected-note@-1  {{Potential leak of memory pointed to by 'data'}}
