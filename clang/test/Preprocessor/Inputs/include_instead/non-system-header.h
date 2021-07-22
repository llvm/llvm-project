#pragma clang include_instead(<include_instead/public1.h>)
// expected-error@-1{{'#pragma clang include_instead' cannot be used outside of system headers}}
