#include <ptrcheck.h>

int *__sized_by(len) alloc_sized_by(int len);
int *alloc_attributed(int len) __attribute__((alloc_size(1)));
