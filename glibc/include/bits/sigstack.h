#include_next <bits/sigstack.h>

#if !defined _ISOMAC && !defined CONSTANT_MINSIGSTKSZ
# define CONSTANT_MINSIGSTKSZ MINSIGSTKSZ
#endif
