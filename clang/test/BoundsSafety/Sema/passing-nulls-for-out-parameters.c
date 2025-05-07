
// XFAIL: *
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// TODO: ended_by
#include <ptrcheck.h>

void out_ptr_out_count(int *__counted_by(*out_count) *out_ptr,
                       unsigned *out_count);

void calls_out_ptr_out_count(void) {
  // with null out buffer and null out count.
  out_ptr_out_count(0, 0);

  // with non-null out buffer and null out count.
  // no error??
  int *__counted_by(4) p1;
  out_ptr_out_count(&p1, 0);

  // with null out buffer and non-null out count.
  // reading `*out_count` is fine in this case, but updating `*out_count`
  // will require updating `*out_ptr` so allowing this is not useful.
  unsigned count1;
  int *__counted_by(count1) p2;
  out_ptr_out_count(0, &count1);

  // with null out buffer and unrelated out count.
  unsigned count2;
  out_ptr_out_count(0, &count2);

  // with incompatible out buffer and null out count.
  // `*out_ptr` will generate promotion to bidi_indexable using `*out_count`,
  // which will lead to null pointer dereference.
  int *__single p3;
  out_ptr_out_count(&p3, 0);

  unsigned count4;
  int *__counted_by(count4) p4;
  out_ptr_out_count(&p4, 0);
}

void out_ptr_in_ptr_out_count(int *__counted_by(*out_count) *out_ptr,
                              int *__counted_by(*out_count) ptr,
                              unsigned *out_count);

void calls_out_ptr_in_ptr_out_count(void) {
  // not allowed due to `ptr`.
  out_ptr_in_ptr_out_count(0, 0, 0);
  int *p1;
  out_ptr_in_ptr_out_count(0, p1, 0);
}

void out_ptr2_out_count(int *__counted_by(*out_count) *out_ptr1,
                        int *__counted_by(*out_count) *out_ptr2,
                        unsigned *out_count);

void calls_out_ptr2_out_count(void) {
  // should be fine
  out_ptr2_out_count(0, 0, 0);

  int *__counted_by(4) p1;
  out_ptr2_out_count(&p1, 0, 0);

  int *__counted_by(0) p2;
  out_ptr2_out_count(0, &p2, 0);

  // doesn't fit in the current model because updating `*out_ptr1` or
  // `*out_count` would require self assignment `*out_ptr2 = *out_ptr2`,
  // which involves null pointer dereference.
  unsigned count3;
  int *__counted_by(count3) p3;
  out_ptr2_out_count(&p3, 0, &count3);

  unsigned count4;
  int *__counted_by(count4) p4;
  out_ptr2_out_count(0, &p4, 0);
}

// ptr promotion to bidi_indexable will access `*out_count`, whether
// `ptr` is  null or not. Thus, `out_count` shouldn't be null.
void in_ptr_out_count(int *__counted_by(*out_count) ptr, unsigned *out_count);

void calls_in_ptr_out_count(void) {
  // no error??
  // with null buffer and null out count.
  in_ptr_out_count(0, 0);

  // with non-null buffer and null out count.
  // no error??
  int *p1;
  in_ptr_out_count(p1, 0);

  // with null buffer and non-null out count.
  unsigned count1;
  int *__counted_by(count1) p2;
  in_ptr_out_count(0, &count1);

  // with null buffer and unrelated out count.
  // no error??
  unsigned count2;
  in_ptr_out_count(0, &count2);
}

// -fbounds-safety doesn't support inout buffer.
void out_ptr_in_count(int *__counted_by(count) *out_ptr, unsigned count);

void calls_out_ptr_in_count(void) {
  // with null buffer and null count.
  // no error?
  in_ptr_out_count(0, 0);

  // with null buffer and non-null count.
  // weird error: non-pointer to safe pointer conversion is not allowed...?
  in_ptr_out_count(0, 2);

  // with non-null buffer and null count.
  int *__counted_by(4) p1;
  in_ptr_out_count(&p1, 0);

  // weird error: non-pointer to safe pointer conversion is not allowed...?
  int *__counted_by(4) p2;
  in_ptr_out_count(&p2, 4);

  // with incompatible, non-null buffer and null count.
  // no error?
  int *__single p3;
  in_ptr_out_count(&p3, 0);

  unsigned count1;
  int *__counted_by(count1) p4;
  in_ptr_out_count(&p4, 0);

  // with null buffer and non-null count.
  // weird error: non-pointer to safe pointer conversion is not allowed...?
  unsigned count2;
  int *__counted_by(count2) p5;
  in_ptr_out_count(0, count2);

  // with null buffer and unrelated out count.
  // weird error: non-pointer to safe pointer conversion is not allowed...?
  unsigned count3;
  in_ptr_out_count(0, count3);
}

void out_ptr_in_ptr_in_count(int *__counted_by(count) *out_ptr,
                             int *__counted_by(count) ptr,
                             unsigned count);

void calls_out_ptr_in_ptr_in_count(void) {
  // should be fine
  out_ptr_in_ptr_in_count(0, 0, 0);

  // should be a warning/error for in ptr
  out_ptr_in_ptr_in_count(0, 0, 10);

  // should be fine
  int arr[10];
  out_ptr_in_ptr_in_count(0, arr, 10);
}
