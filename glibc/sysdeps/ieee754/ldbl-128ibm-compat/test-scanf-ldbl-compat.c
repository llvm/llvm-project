/* Include stdio.h from libio/, because include/stdio.h and -std=c89 do
   not work together.  */
#include <libio/stdio.h>

#define CHAR char
#define Lx(x) x
#define L(x) Lx (x)
#define FSCANF fscanf
#define SSCANF sscanf
#define SCANF scanf
#define VFSCANF vfscanf
#define VSSCANF vsscanf
#define VSCANF vscanf
#define STRCPY strcpy
#include <test-scanf-ldbl-compat-template.c>
