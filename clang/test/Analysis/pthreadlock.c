// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.unix.PthreadLock -verify %s

// Tests performing normal locking patterns and wrong locking orders

typedef struct {
	void	*foo;
} pthread_mutex_t;

typedef pthread_mutex_t lck_mtx_t;

extern int pthread_mutex_lock(pthread_mutex_t *);
extern int pthread_mutex_unlock(pthread_mutex_t *);
extern int pthread_mutex_trylock(pthread_mutex_t *);
extern int lck_mtx_lock(lck_mtx_t *);
extern int lck_mtx_unlock(lck_mtx_t *);
extern int lck_mtx_try_lock(lck_mtx_t *);

pthread_mutex_t mtx1, mtx2;
lck_mtx_t lck1, lck2;

void
ok1(void)
{
	pthread_mutex_lock(&mtx1); // no-warning
}

void
ok2(void)
{
	pthread_mutex_unlock(&mtx1); // no-warning
}

void
ok3(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
}

void
ok4(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
}

void
ok5(void)
{
	if (pthread_mutex_trylock(&mtx1) == 0)	// no-warning
		pthread_mutex_unlock(&mtx1);	// no-warning
}

void
ok6(void)
{
	lck_mtx_lock(&lck1);		// no-warning
}

void
ok7(void)
{
	if (lck_mtx_try_lock(&lck1) != 0)	// no-warning
		lck_mtx_unlock(&lck1);		// no-warning
}

void
bad1(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// expected-warning{{This lock has already been acquired}}
}

void
bad2(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// expected-warning{{This lock has already been acquired}}
}

void
bad3(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// expected-warning{{This was not the most recently acquired lock}}
	pthread_mutex_unlock(&mtx2);
}

void
bad4(void)
{
	if (pthread_mutex_trylock(&mtx1)) // no-warning
		return;
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// expected-warning{{This was not the most recently acquired lock}}
}

void
bad5(void)
{
	lck_mtx_lock(&lck1);	// no-warning
	lck_mtx_lock(&lck1);	// expected-warning{{This lock has already been acquired}}
}

void
bad6(void)
{
	lck_mtx_lock(&lck1);	// no-warning
	lck_mtx_unlock(&lck1);	// no-warning
	lck_mtx_lock(&lck1);	// no-warning
	lck_mtx_lock(&lck1);	// expected-warning{{This lock has already been acquired}}
}

void
bad7(void)
{
	lck_mtx_lock(&lck1);	// no-warning
	lck_mtx_lock(&lck2);	// no-warning
	lck_mtx_unlock(&lck1);	// expected-warning{{This was not the most recently acquired lock}}
	lck_mtx_unlock(&lck2);
}

void
bad8(void)
{
	if (lck_mtx_try_lock(&lck1) == 0) // no-warning
		return;
	lck_mtx_lock(&lck2);		// no-warning
	lck_mtx_unlock(&lck1);		// expected-warning{{This was not the most recently acquired lock}}
}
