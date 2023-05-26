#pragma clang system_header

int abort();

#ifdef NDEBUG
#define assert(x) 1
#else
#define assert(x)                                                              \
  if (!(x))                                                                    \
  (void)abort()
#endif

void print(...);
#define assert2(e) (__builtin_expect(!(e), 0) ?                                \
                       print (#e, __FILE__, __LINE__) : (void)0)

#ifdef NDEBUG
#define my_assert(x) 1
#else
#define my_assert(x)                                                           \
  ((void)((x) ? 1 : abort()))
#endif

#ifdef NDEBUG
#define not_my_assert(x) 1
#else
#define not_my_assert(x)                                                       \
  if (!(x))                                                                    \
  (void)abort()
#endif

#define real_assert(x) ((void)((x) ? 1 : abort()))
#define wrap1(x) real_assert(x)
#define wrap2(x) wrap1(x)
#define convoluted_assert(x) wrap2(x)

#define msvc_assert(expression) (void)(                                        \
            (!!(expression)) ||                                                \
            (abort(), 0)                                                       \
        )
