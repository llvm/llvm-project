#include <sysdep.h>

int __pthread_enable_asynccancel (void);
void __pthread_disable_asynccancel (int oldtype);

#pragma weak __pthread_enable_asynccancel
#pragma weak __pthread_disable_asynccancel

/* Always multi-thread (since there's at least the sig handler), but no
   handling enabled.  */
#define SINGLE_THREAD_P (0)
#define RTLD_SINGLE_THREAD_P (0)

#define LIBC_CANCEL_ASYNC() ({ \
	int __cancel_oldtype = 0; \
	if (__pthread_enable_asynccancel) \
		__cancel_oldtype = __pthread_enable_asynccancel(); \
	__cancel_oldtype; \
})

#define LIBC_CANCEL_RESET(val) do { \
	if (__pthread_disable_asynccancel) \
		__pthread_disable_asynccancel (val); \
} while (0)
