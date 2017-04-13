// RUN: %clang_asan -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NOSCRIBBLE %s
// RUN: env MallocScribble=1 MallocPreScribble=1 %run %t 2>&1 | FileCheck --check-prefix=CHECK-SCRIBBLE %s
// RUN: %env_asan_opts=max_free_fill_size=4096 %run %t 2>&1 | FileCheck --check-prefix=CHECK-SCRIBBLE %s

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Isa {
  const char *class_name;
};

struct MyClass {
  long padding;
  Isa *isa;
  long data;

  void print_my_class_name();
};

__attribute__((no_sanitize("address")))
void MyClass::print_my_class_name() {
  fprintf(stderr, "this = %p\n", this);
  fprintf(stderr, "padding = 0x%lx\n", this->padding);
  fprintf(stderr, "isa = %p\n", this->isa);

  if ((uint32_t)(uintptr_t)this->isa != 0x55555555) {
    fprintf(stderr, "class name: %s\n", this->isa->class_name);
  }
}

int main() {
  Isa *my_class_isa = (Isa *)malloc(sizeof(Isa));
  memset(my_class_isa, 0x77, sizeof(Isa));
  my_class_isa->class_name = "MyClass";

  MyClass *my_object = (MyClass *)malloc(sizeof(MyClass));
  memset(my_object, 0x88, sizeof(MyClass));
  my_object->isa = my_class_isa;
  my_object->data = 42;

  my_object->print_my_class_name();
  // CHECK-SCRIBBLE: class name: MyClass
  // CHECK-NOSCRIBBLE: class name: MyClass

  free(my_object);

  my_object->print_my_class_name();
  // CHECK-NOSCRIBBLE: class name: MyClass
  // CHECK-SCRIBBLE: isa = {{(0x)?}}{{5555555555555555|55555555}}

  fprintf(stderr, "okthxbai!\n");
  // CHECK-SCRIBBLE: okthxbai!
  // CHECK-NOSCRIBBLE: okthxbai!
  free(my_class_isa);
}
