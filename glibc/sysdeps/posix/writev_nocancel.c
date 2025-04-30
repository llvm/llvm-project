#include <not-cancel.h>
#define __writev __writev_nocancel
#define __write __write_nocancel
#include <sysdeps/posix/writev.c>
