#ifndef HEADER_LIB_H
#define HEADER_LIB_H

void *custom_realloc(void *member, unsigned size);

int *global_int;

int unavailable_function(void);
int unavailable_global_int;

void do_something_with_pointers(int *ptr1, int *ptr2);
void do_something_with_arrays(int simple[], int nested[][2]);

typedef int unavailable_typedef;
struct unavailable_struct { int x, y, z; };

void take_pointer_and_int(int *ptr1, int value);

#endif
