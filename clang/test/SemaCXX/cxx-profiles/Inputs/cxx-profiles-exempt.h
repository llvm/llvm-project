// Exempted by [[profiles::exempt]], together with the headers it includes.
#include "cxx-profiles-exempt-nested.h"
inline char *exempt_cast(int *p) { return reinterpret_cast<char *>(p); }
