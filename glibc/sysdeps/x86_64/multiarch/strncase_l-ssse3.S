#define USE_SSSE3 1
#define USE_AS_STRNCASECMP_L
#define NO_NOLOCALE_ALIAS
#define STRCMP __strncasecmp_l_ssse3
#define __strncasecmp __strncasecmp_ssse3
#include "../strcmp.S"
