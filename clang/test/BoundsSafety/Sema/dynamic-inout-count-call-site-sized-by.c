
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

/* -------------------
         sized_by
   ------------------- */
struct foo {
  int *__sized_by(size) p;
  int *__sized_by(size) q;
  int size;
};

void in_buf_inout_size(int *__sized_by(*size) buf, int *size);

void inout_buf_inout_size(int *__sized_by(*size) * buf, int *size) {
  inout_buf_inout_size(buf, size);

  in_buf_inout_size(*buf, size);
}

void in_buf_inout_size(int *__sized_by(*size) buf, int *size) {
  in_buf_inout_size(buf, size);

  // expected-error@+1{{parameter 'buf' with '__sized_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_buf_inout_size(&buf, size);
}

void in_buf_inout_buf_inout_size(int *__sized_by(*size) p, int *__sized_by(*size) * q, int *size) {
  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'q' because the type of 'q', 'int *__single __sized_by(*size)*__single' (aka 'int *__single*__single'), refers to 'size'}}
  in_buf_inout_size(p, size);

  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by(*size)' (aka 'int *__single'), refers to 'size'}}
  inout_buf_inout_size(q, size);

  // expected-error@+1{{parameter 'p' with '__sized_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  in_buf_inout_buf_inout_size(*q, &p, size);
}

void bar(void) {
  int array[3] = {1, 2, 3};

  int size = 12;
  int *__sized_by(size) p = array;

  int non_related_size = 12;

  struct foo f;
  f.size = 12;
  f.p = array;
  f.q = array;

  in_buf_inout_size(p, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size(0, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'q' or its address because the type of 'q', 'int *__single __sized_by(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size(f.p, &f.size);

  in_buf_inout_buf_inout_size(f.p, &f.q, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size(array, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size(array, &size);

  in_buf_inout_size(array, &non_related_size);

  in_buf_inout_size(f.p, &non_related_size);

}

/* -------------------
     sized_by_or_null
   ------------------- */
struct foo_nullable {
  int *__sized_by_or_null(size) p;
  int *__sized_by_or_null(size) q;
  int size;
};

void in_buf_inout_size_nullable(int *__sized_by_or_null(*size) buf, int *size);

void inout_buf_inout_size_nullable(int *__sized_by_or_null(*size) * buf, int *size) {
  inout_buf_inout_size_nullable(buf, size);

  in_buf_inout_size_nullable(*buf, size);
}

void in_buf_inout_size_nullable(int *__sized_by_or_null(*size) buf, int *size) {
  in_buf_inout_size_nullable(buf, size);

  // expected-error@+1{{parameter 'buf' with '__sized_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_buf_inout_size_nullable(&buf, size);
}

void in_buf_inout_buf_inout_size_nullable(int *__sized_by_or_null(*size) p, int *__sized_by_or_null(*size) * q, int *size) {
  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'q' because the type of 'q', 'int *__single __sized_by_or_null(*size)*__single' (aka 'int *__single*__single'), refers to 'size'}}
  in_buf_inout_size_nullable(p, size);

  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by_or_null(*size)' (aka 'int *__single'), refers to 'size'}}
  inout_buf_inout_size_nullable(q, size);

  // expected-error@+1{{parameter 'p' with '__sized_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  in_buf_inout_buf_inout_size_nullable(*q, &p, size);
}

void bar_nullable(void) {
  int array[3] = {1, 2, 3};

  int size = 12;
  int *__sized_by_or_null(size) p = array;

  int non_related_size = 12;

  struct foo_nullable f;
  f.size = 12;
  f.p = array;
  f.q = array;

  in_buf_inout_size_nullable(p, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(0, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'q' or its address because the type of 'q', 'int *__single __sized_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(f.p, &f.size);

  in_buf_inout_buf_inout_size_nullable(f.p, &f.q, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(array, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __sized_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(array, &size);

  in_buf_inout_size_nullable(array, &non_related_size);

  in_buf_inout_size_nullable(f.p, &non_related_size);

}
