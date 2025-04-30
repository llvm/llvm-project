#include <wchar.h>
#define CHAR wchar_t
#define L(str) L##str
#define SSCANF swscanf

#include "bug18.c"
