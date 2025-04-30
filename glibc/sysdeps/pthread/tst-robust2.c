#define AFTER_JOIN 1
#define LOCK(m) pthread_mutex_trylock (m)
#include "tst-robust1.c"
