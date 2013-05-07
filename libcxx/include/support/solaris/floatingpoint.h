#define atof sun_atof
#define strtod sun_strtod
#include_next "floatingpoint.h"
#undef atof
#undef strtod
