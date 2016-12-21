// Simply marking this as "#pragma clang system_header" didn't tickle the bug, rdar://problem/21134250.

void system1(int *ptr);
#if WARN_IN_SYSTEM_HEADERS
// expected-warning@-2{{pointer is missing a nullability type specifier}}
#endif

void system2(int * _Nonnull);
