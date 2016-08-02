#ifndef HEADER_LIB_H
#define HEADER_LIB_H

void *custom_realloc(void *member, unsigned size);

int *global_int;

int unavailable_function(void);
int unavailable_global_int;

void do_something_with_pointers(int *ptr1, int *ptr2);

#endif
