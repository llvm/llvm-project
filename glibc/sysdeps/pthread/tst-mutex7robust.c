/* Bug 21778: Fix oversight in robust mutex lock acquisition.  */
#define TYPE PTHREAD_MUTEX_NORMAL
#define ROBUST PTHREAD_MUTEX_ROBUST
#define DELAY_NSEC 0
#define ROUNDS 1000
#define N 32
#include "tst-mutex7.c"
