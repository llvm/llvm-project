#ifdef SHARED
# define USE_AS_STRNCMP
# define STRCMP	__strncmp_ssse3
# include "strcmp-ssse3.S"
#endif
