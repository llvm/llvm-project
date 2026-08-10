#if defined(CLANG)
#pragma clang system_header
// expected-no-diagnostics
#elif defined(GCC)
#pragma GCC system_header
// expected-no-diagnostics
#elif defined(MS)
#pragma system_header
// expected-no-diagnostics
#else
// expected-warning@+1{{unknown pragma ignored}}
#pragma system_header

// expected-note@+4{{previous definition is here}}
// expected-warning@+4{{redefinition of typedef 'x' is a C11 feature}}
#endif

typedef int x;
typedef int x;
