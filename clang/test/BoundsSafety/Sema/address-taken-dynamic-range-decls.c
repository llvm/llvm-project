
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

struct S {
  int *__ended_by(end) start;
  int *end;
};

struct U {
  int *__ended_by(iter) start;
  int *__ended_by(end) iter;
  int *end;
};

void fun_out_end(int **out_end, int *__ended_by(*out_end) * out_start);
void fun_out_seq(int **out_end,
                 int *__ended_by(*out_iter) * out_start,
                 int *__ended_by(*out_end) * out_iter);

void fun_in_start_out_end(int *__ended_by(*out_end) start, int **out_end) {
  // expected-error@+1{{parameter 'start' with '__ended_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  fun_out_end(out_end, &start);
}

void fun_no_out_end(int **no_out_end, int **no_out_start);

void test() {
  struct S s = {0};
  struct U u = {0};

  int *__single local_end = 0;
  int **__single ptr_to_end = &s.end; // expected-error{{field referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **ptr_to_start = &s.start; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int ***ptr_ptr_to_end = &ptr_to_end;
  ptr_to_end = &s.end;  // expected-error{{field referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  *ptr_to_start = &s.start; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  struct S *struct_ptr = &s;
  *ptr_to_end = &struct_ptr->end;   // expected-error{{field referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  *ptr_to_end = &(*struct_ptr).end; // expected-error{{field referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}

  *ptr_to_end = 0;

  fun_out_end(&s.end, &s.start);
  // expected-error@+1{{type of 'local_end', 'int *__single', is incompatible with parameter of type 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single')}}
  fun_out_end(&local_end, &s.start);
  // expected-error@+1{{passing address of 'end' as an indirect parameter; must also pass 'iter' or its address because the type of 'iter', 'int *__single __ended_by(end) /* __started_by(start) */ ' (aka 'int *__single'), refers to 'end'}}
  fun_out_end(&u.end, &u.start);
  fun_out_seq(&u.end, &u.start, &u.iter);
  // expected-error@+1{{type of 'start', 'int *__single __ended_by(iter)' (aka 'int *__single'), is incompatible with parameter of type 'int *__single /* __started_by(*out_iter) */ ' (aka 'int *__single')}}
  fun_out_seq(&u.start, &u.iter, &u.end);
  // expected-error@+1{{passing address of 'start' as an indirect parameter; must also pass 'iter' or its address because the type of 'start', 'int *__single __ended_by(iter)' (aka 'int *__single'), refers to 'iter'}}
  fun_out_seq(&s.end, &u.start, &s.start);
  // expected-error@+1{{type of 'end', 'int *__single /* __started_by(start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single*__single'}}
  fun_no_out_end(&s.end, &s.start);
}
