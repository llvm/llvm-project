/* Include stdio.h from libio/, because include/stdio.h and -std=c89 do
   not work together.  */
#include <libio/stdio.h>

#define CHAR wchar_t
#define Lx(x) L##x
#define L(x) Lx (x)
#define FSCANF fwscanf
#define SSCANF swscanf
#define SCANF wscanf
#define VFSCANF vfwscanf
#define VSSCANF vswscanf
#define VSCANF vwscanf
#define STRCPY wcscpy
#include <test-scanf-ldbl-compat-template.c>
