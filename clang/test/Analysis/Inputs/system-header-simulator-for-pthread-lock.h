// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, pthread_mutex* functions
// should not invalidate regions of their arguments.
#pragma clang system_header

typedef struct {
	void	*foo;
} pthread_mutex_t;

typedef struct {
	void	*foo;
} pthread_mutexattr_t;

typedef struct {
	void	*foo;
} lck_grp_t;

typedef struct {
  void *foo;
} lck_rw_t;

typedef pthread_mutex_t lck_mtx_t;

extern void fake_system_function();
extern void fake_system_function_that_takes_a_mutex(pthread_mutex_t *mtx);

extern int pthread_mutex_lock(pthread_mutex_t *);
extern int pthread_mutex_unlock(pthread_mutex_t *);
extern int pthread_mutex_trylock(pthread_mutex_t *);
extern int pthread_mutex_destroy(pthread_mutex_t *);
extern int pthread_mutex_init(pthread_mutex_t  *mutex, const pthread_mutexattr_t *mutexattr);

typedef int boolean_t;
extern void lck_mtx_lock(lck_mtx_t *);
extern void lck_mtx_unlock(lck_mtx_t *);
extern boolean_t lck_mtx_try_lock(lck_mtx_t *);
extern void lck_mtx_destroy(lck_mtx_t *lck, lck_grp_t *grp);

extern void lck_rw_lock_exclusive(lck_rw_t *lck);
extern void lck_rw_unlock_exclusive(lck_rw_t *lck);
extern void lck_rw_lock_shared(lck_rw_t *lck);
extern void lck_rw_unlock_shared(lck_rw_t *lck);
