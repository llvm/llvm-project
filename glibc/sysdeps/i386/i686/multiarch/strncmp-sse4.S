#ifdef SHARED
# define USE_AS_STRNCMP
# define STRCMP	__strncmp_sse4_2
# include "strcmp-sse4.S"
#endif
