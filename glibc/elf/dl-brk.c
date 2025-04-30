/* We can use the normal code but we also know the __curbrk is not exported
   from ld.so.  */
extern void *__curbrk attribute_hidden;

#include <brk.c>
