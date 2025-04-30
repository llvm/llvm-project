#include <wchar.h>
#define CHAR wchar_t
#define L(str) L##str
#define FPUTS fputws
#define FSCANF fwscanf

#include "bug19.c"
