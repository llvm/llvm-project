#pragma clang system_header

// Define 'errno' as a macro that calls a function.
int *__errno_location();
#define errno (*__errno_location())
