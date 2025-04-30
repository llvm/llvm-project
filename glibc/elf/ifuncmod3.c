/* Test STT_GNU_IFUNC symbols with dlopen.  */

#include "ifuncmod1.c"

int ret_foo;
int ret_foo_hidden;
int ret_foo_protected;
