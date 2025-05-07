
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

/* -------------------
        counted_by
   ------------------- */
struct foo {
  int *__counted_by(len) p;
  int *__counted_by(len) q;
  int len;
};

void in_buf_inout_len(int *__counted_by(*len) buf, int *len);

void inout_buf_inout_len(int *__counted_by(*len) * buf, int *len) {
  inout_buf_inout_len(buf, len);

  in_buf_inout_len(*buf, len);
}

void in_buf_inout_len(int *__counted_by(*len) buf, int *len) {
  in_buf_inout_len(buf, len);

  // expected-error@+1{{parameter 'buf' with '__counted_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_buf_inout_len(&buf, len);
}

void in_buf_inout_buf_inout_len(int *__counted_by(*len) p, int *__counted_by(*len) * q, int *len) {
  // expected-error@+1{{passing 'len' as an indirect parameter; must also pass 'q' because the type of 'q', 'int *__single __counted_by(*len)*__single' (aka 'int *__single*__single'), refers to 'len'}}
  in_buf_inout_len(p, len);

  // expected-error@+1{{passing 'len' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by(*len)' (aka 'int *__single'), refers to 'len'}}
  inout_buf_inout_len(q, len);

  // expected-error@+1{{parameter 'p' with '__counted_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  in_buf_inout_buf_inout_len(*q, &p, len);
}

void bar(void) {
  int array[3] = {1, 2, 3};

  int len = 3;
  int *__counted_by(len) p = array;

  int non_related_len = 3;

  struct foo f;
  f.len = 3;
  f.p = array;
  f.q = array;

  in_buf_inout_len(p, &len);

  // expected-error@+1{{passing address of 'len' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  in_buf_inout_len(0, &len);

  // expected-error@+1{{passing address of 'len' as an indirect parameter; must also pass 'q' or its address because the type of 'q', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  in_buf_inout_len(f.p, &f.len);

  in_buf_inout_buf_inout_len(f.p, &f.q, &f.len);

  // expected-error@+1{{passing address of 'len' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  in_buf_inout_len(array, &f.len);

  // expected-error@+1{{passing address of 'len' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  in_buf_inout_len(array, &len);

  in_buf_inout_len(array, &non_related_len);

  in_buf_inout_len(f.p, &non_related_len);
}

/* -------------------
   counted_by_or_null
   ------------------- */
struct foo_nullable {
  int *__counted_by_or_null(size) p;
  int *__counted_by_or_null(size) q;
  int size;
};

void in_buf_inout_size_nullable(int *__counted_by_or_null(*size) buf, int *size);

void inout_buf_inout_size_nullable(int *__counted_by_or_null(*size) * buf, int *size) {
  inout_buf_inout_size_nullable(buf, size);

  in_buf_inout_size_nullable(*buf, size);
}

void in_buf_inout_size_nullable(int *__counted_by_or_null(*size) buf, int *size) {
  in_buf_inout_size_nullable(buf, size);

  // expected-error@+1{{parameter 'buf' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_buf_inout_size_nullable(&buf, size);
}

void in_buf_inout_buf_inout_size_nullable(int *__counted_by_or_null(*size) p, int *__counted_by_or_null(*size) * q, int *size) {
  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'q' because the type of 'q', 'int *__single __counted_by_or_null(*size)*__single' (aka 'int *__single*__single'), refers to 'size'}}
  in_buf_inout_size_nullable(p, size);

  // expected-error@+1{{passing 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by_or_null(*size)' (aka 'int *__single'), refers to 'size'}}
  inout_buf_inout_size_nullable(q, size);

  // expected-error@+1{{parameter 'p' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  in_buf_inout_buf_inout_size_nullable(*q, &p, size);
}

void bar_nullable(void) {
  int array[3] = {1, 2, 3};

  int size = 3;
  int *__counted_by_or_null(size) p = array;

  int non_related_size = 12;

  struct foo_nullable f;
  f.size = 3;
  f.p = array;
  f.q = array;

  in_buf_inout_size_nullable(p, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(0, &size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'q' or its address because the type of 'q', 'int *__single __counted_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(f.p, &f.size);

  in_buf_inout_buf_inout_size_nullable(f.p, &f.q, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(array, &f.size);

  // expected-error@+1{{passing address of 'size' as an indirect parameter; must also pass 'p' or its address because the type of 'p', 'int *__single __counted_by_or_null(size)' (aka 'int *__single'), refers to 'size'}}
  in_buf_inout_size_nullable(array, &size);

  in_buf_inout_size_nullable(array, &non_related_size);

  in_buf_inout_size_nullable(f.p, &non_related_size);

}
