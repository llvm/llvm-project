

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null

#include <ptrcheck.h>

struct S {
    int *__counted_by(count) ptr;
    int count;
};

int assign_count_only_never_succeeds(struct S *s) {
    s->count += 1;
    s->ptr = s->ptr;
}

int assign_count_only(struct S *s) {
    s->count -= 1;
    s->ptr = s->ptr;
}

int assign_ptr(struct S *s) {
    s->count = s->count - 1;
    s->ptr = s->ptr + 1;
}
