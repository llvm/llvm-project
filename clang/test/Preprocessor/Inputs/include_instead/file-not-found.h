#pragma GCC system_header
#pragma clang include_instead(<include_instead/does_not_exist.h>)
// expected-error@-1{{'include_instead/does_not_exist.h' file not found}}
