#define STRCMP __strncasecmp_l
#define USE_AS_STRNCASECMP_L
#include "strcmp.S"

weak_alias (__strncasecmp_l, strncasecmp_l)
libc_hidden_def (strncasecmp_l)
