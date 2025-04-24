// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,rs %s
#include <ptrcheck.h>

int *__ended_by(end) func_ret_end(int *end) {
  // rs-error@+1{{parameter 'end' is implicitly read-only due to being used by the '__ended_by' attribute in the return type of 'func_ret_end' ('int *__single __ended_by(end)' (aka 'int *__single'))}}
  end = 0;
  return end;
}

int *__ended_by(end) func_ret_end2(int *__ended_by(end) start, int *end) {
  return start + 1;
}

void func_out_start_out_end(int *__ended_by(*out_end) *out_start,
                            int **out_end) {
  out_start = 0;
  out_end = 0;
}

void func_out_start_in_end(int *__ended_by(end) *out_start,
                           int *end) {
  *out_start = *out_start + 1;
  // expected-error@+1{{parameter 'end' referred to by an indirect '__ended_by' pointer is implicitly read-only}}
  end--;
}

void func_out_start_in_end2(int *__ended_by(end) *out_start,
                            int *end) {
  // expected-error@+1{{parameter 'end' referred to by an indirect '__ended_by' pointer is implicitly read-only}}
  end--;
}

void func_out_start_in_end3(int *__ended_by(end) *out_start,
                            int *end) {
  *out_start = *out_start + 1;
}

void func_in_start_out_end(int *__ended_by(*out_end) start,
                           int **out_end) {
  // expected-error@+1{{parameter 'start' with '__ended_by' attribute depending on an indirect end pointer is implicitly read-only}}
  start++;
}

void func_in_start_out_end2(int *__ended_by(*out_end) start,
                            int **out_end) {
  *out_end = *out_end - 1;
  // expected-error@+1{{parameter 'start' with '__ended_by' attribute depending on an indirect end pointer is implicitly read-only}}
  start++;
}

void func_in_start_out_end3(int *__ended_by(*out_end) start,
                            int **out_end) {
  *out_end = *out_end - 1;
}
