// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,rs %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,rs %s

#include <ptrcheck.h>

void i(int);
void pi(int *);
void ppi(int **);
void pppi(int ***);

// __counted_by()

void cb_in_in(int count, int *__counted_by(count) ptr);
void cb_in_out(int count, int *__counted_by(count) *ptr);
void cb_out_in(int *count, int *__counted_by(*count) ptr);
void cb_out_out(int *count, int *__counted_by(*count) *ptr);

int *__counted_by(count) ret_cb_in(int count);
int *__counted_by(*count) ret_cb_out(int *count);

int *__counted_by(count) mixed_simple(int count, int *__counted_by(count) ptr);
int *__counted_by(*count) mixed_inout_count(int *count, int *__counted_by(*count) ptr);
int *__counted_by(*count) mixed_out_ptr(int *count, int *__counted_by(*count) *ptr);
int *__counted_by(*count) mixed_inout_count_out_ptr(int *count, int *__counted_by(*count) ptr, int *__counted_by(*count) *out_ptr);

int *__counted_by(count) test_cb_in(int count) {
  int c;
  int *__counted_by(c) p;

  count = 42; // rs-error{{parameter 'count' is implicitly read-only due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}

  i(count);
  pi(&count); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}

  cb_in_in(count, p);
  cb_in_out(count, &p);   // expected-error{{passing address of 'p' as an indirect parameter; must also pass 'c' or its address because the type of 'p', 'int *__single __counted_by(c)' (aka 'int *__single'), refers to 'c'}}
  cb_out_in(&count, p);   // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  cb_out_out(&count, &p); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}

  (void)ret_cb_in(count);
  (void)ret_cb_out(&count); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}

  (void)mixed_simple(count, p);
  (void)mixed_inout_count(&count, p);             // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  (void)mixed_out_ptr(&count, &p);                // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  (void)mixed_inout_count_out_ptr(&count, p, &p); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_in' ('int *__single __counted_by(count)' (aka 'int *__single'))}}
}

int *__counted_by(*count) test_cb_out(int *count) {
  int c;
  int *__counted_by(c) p;

  *count = 42;
  count = p; // expected-error{{not allowed to change out parameter used as dependent count expression of other parameter or return type}}

  i(*count);
  pi(count);   // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_out' ('int *__single __counted_by(*count)' (aka 'int *__single'))}}
  ppi(&count); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_out' ('int *__single __counted_by(*count)' (aka 'int *__single'))}}

  cb_in_in(*count, p);
  cb_in_out(*count, &p); // expected-error{{passing address of 'p' as an indirect parameter; must also pass 'c' or its address because the type of 'p', 'int *__single __counted_by(c)' (aka 'int *__single'), refers to 'c'}}
  cb_out_in(count, p);
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  cb_out_out(count, &p); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_out' ('int *__single __counted_by(*count)' (aka 'int *__single'))}}

  (void)ret_cb_in(*count);
  (void)ret_cb_out(count);

  (void)mixed_simple(*count, p);
  (void)mixed_inout_count(count, p);
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  (void)mixed_out_ptr(count, &p);                // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_out' ('int *__single __counted_by(*count)' (aka 'int *__single'))}}
  // legacy-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single'}}
  (void)mixed_inout_count_out_ptr(count, p, &p); // rs-error{{parameter 'count' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__counted_by' attribute in the return type of 'test_cb_out' ('int *__single __counted_by(*count)' (aka 'int *__single'))}}
}

// Other variants of __counted_by()
// The logic is the same as for __counted_by(), so test only a few cases.

int *__counted_by_or_null(count) test_cbn_in(int count) {
  count = 42; // rs-error{{parameter 'count' is implicitly read-only due to being used by the '__counted_by_or_null' attribute in the return type of 'test_cbn_in' ('int *__single __counted_by_or_null(count)' (aka 'int *__single'))}}
}

void *__sized_by(size) test_sb_in(int size) {
  size = 42; // rs-error{{parameter 'size' is implicitly read-only due to being used by the '__sized_by' attribute in the return type of 'test_sb_in' ('void *__single __sized_by(size)' (aka 'void *__single'))}}
}

void *__sized_by_or_null(size) test_sbn_in(int size) {
  size = 42; // rs-error{{parameter 'size' is implicitly read-only due to being used by the '__sized_by_or_null' attribute in the return type of 'test_sbn_in' ('void *__single __sized_by_or_null(size)' (aka 'void *__single'))}}
}

// __ended_by()

void eb_in(int *__ended_by(end) start, int *end);
void eb_out(int *__ended_by(*end) start, int **end);

int *__ended_by(end) ret_eb_in(int *end);
int *__ended_by(*end) ret_eb_out(int **end);

int *__ended_by(end) test_eb_in(int *end) {
  int *p;

  *end = 42;
  end = p; // rs-error{{parameter 'end' is implicitly read-only due to being used by the '__ended_by' attribute in the return type of 'test_eb_in' ('int *__single __ended_by(end)' (aka 'int *__single'))}}

  i(*end);
  pi(end);
  ppi(&end); // rs-error{{parameter 'end' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__ended_by' attribute in the return type of 'test_eb_in' ('int *__single __ended_by(end)' (aka 'int *__single'))}}

  eb_in(p, end);
  eb_out(p, &end); // expected-error{{type of 'end', 'int *__single', is incompatible with parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single')}}

  (void)ret_eb_in(end);
  (void)ret_eb_out(&end); // rs-error{{parameter 'end' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__ended_by' attribute in the return type of 'test_eb_in' ('int *__single __ended_by(end)' (aka 'int *__single'))}}
}

int *__ended_by(*end) test_eb_out(int **end) {
  int *__single p;

  **end = 42;
  *end = p;
  end = &p; // rs-error{{parameter 'end' is implicitly read-only due to being used by the '__ended_by' attribute in the return type of 'test_eb_out' ('int *__single __ended_by(*end)' (aka 'int *__single'))}}

  i(**end);
  pi(*end);
  ppi(end);   // rs-error{{parameter 'end' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__ended_by' attribute in the return type of 'test_eb_out' ('int *__single __ended_by(*end)' (aka 'int *__single'))}}
  pppi(&end); // rs-error{{parameter 'end' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__ended_by' attribute in the return type of 'test_eb_out' ('int *__single __ended_by(*end)' (aka 'int *__single'))}}

  eb_in(p, *end);
  eb_out(p, end); // expected-error{{type of 'end', 'int *__single*__single', is incompatible with parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single')}}

  (void)ret_eb_in(*end);
  (void)ret_eb_out(end); // rs-error{{parameter 'end' is implicitly read-only and cannot be passed as an indirect argument due to being used by the '__ended_by' attribute in the return type of 'test_eb_out' ('int *__single __ended_by(*end)' (aka 'int *__single'))}}
}
