// RUN: %clang_cc1 -triple x86_64-unknown-linux -ast-print %s -o - | FileCheck %s

typedef unsigned long int size_t;

int global_array[42];
int global_int;

struct fam_struct {
  int x;
  char count;
  int array[] __attribute__((counted_by(count)));
};

// CHECK-LABEL: void test1(struct fam_struct *ptr, int size) {
// CHECK-NEXT:      size_t __ignored_assignment;
// CHECK-NEXT:      *_Generic(__builtin_counted_by_ref(ptr->array), void *: &__ignored_assignment, default: __builtin_counted_by_ref(ptr->array)) = 42;
void test1(struct fam_struct *ptr, int size) {
  size_t __ignored_assignment;

  *_Generic(__builtin_counted_by_ref(ptr->array),
           void *: &__ignored_assignment,
           default: __builtin_counted_by_ref(ptr->array)) = 42; // ok
}
