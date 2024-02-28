// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.

#pragma clang system_header

#ifdef __cplusplus
#define restrict /*restrict*/
#endif

typedef __builtin_va_list va_list;

#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#define va_copy(dst, src)   __builtin_va_copy(dst, src)

int vprintf (const char *restrict format, va_list arg);

int vsprintf (char *restrict s, const char *restrict format, va_list arg);

int some_library_function(int n, va_list arg);

// No warning from system header.
inline void __impl_detail(int fst, ...) {
  va_list va;
  (void)va_arg(va, int);
}
