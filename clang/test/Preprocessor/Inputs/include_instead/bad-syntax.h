#pragma GCC system_header

#pragma clang include_instead <include_instead/public-before.h>
// expected-error@-1{{expected (}}

#pragma clang include_instead(<include_instead/public-after.h>]
// expected-error@-1{{expected )}}
