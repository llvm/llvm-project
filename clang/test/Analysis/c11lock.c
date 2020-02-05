// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.C11Lock -verify %s

typedef int mtx_t;
struct timespec;

enum {
  // FIXME: The value if this enum is implementation defined. While all the
  // implementations I am aware of using 0, the right solution would be to
  // look this value up in the AST (and disable the check if it is not found).
  thrd_success = 0,
  thrd_error = 2
};

int mtx_init(mtx_t *mutex, int type);
int mtx_lock(mtx_t *mutex);
int mtx_timedlock(mtx_t *mutex,
                  const struct timespec *time_point);
int mtx_trylock(mtx_t *mutex);
int mtx_unlock(mtx_t *mutex);
int mtx_destroy(mtx_t *mutex);

mtx_t mtx1;
mtx_t mtx2;

void bad1(void)
{
  mtx_lock(&mtx1);	// no-warning
  mtx_lock(&mtx1);	// expected-warning{{This lock has already been acquired}}
}

void bad2(void) {
  mtx_t mtx;
  mtx_init(&mtx, 0);
  mtx_lock(&mtx);
} // TODO: Warn for missing unlock?

void bad3(void) {
  mtx_t mtx;
  mtx_init(&mtx, 0);
} // TODO: Warn for missing destroy?

void bad4(void) {
  mtx_t mtx;
  mtx_init(&mtx, 0);
  mtx_lock(&mtx);
  mtx_unlock(&mtx);
} // TODO: warn for missing destroy?

void bad5(void) {
  mtx_lock(&mtx1);
  mtx_unlock(&mtx1);
  mtx_unlock(&mtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad6() {
  mtx_init(&mtx1, 0);
  if (mtx_trylock(&mtx1) != thrd_success)
    mtx_unlock(&mtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad7(void) {
  mtx_lock(&mtx1);
  mtx_lock(&mtx2);
  mtx_unlock(&mtx1); // expected-warning {{This was not the most recently acquired lock. Possible lock order reversal}}
  mtx_unlock(&mtx2);
}

void good() {
  mtx_t mtx;
  mtx_init(&mtx, 0);
  mtx_lock(&mtx);
  mtx_unlock(&mtx);
  mtx_destroy(&mtx);
}

void good2() {
  mtx_t mtx;
  mtx_init(&mtx, 0);
  if (mtx_trylock(&mtx) == thrd_success)
    mtx_unlock(&mtx);
  mtx_destroy(&mtx);
}

void good3() {
  mtx_t mtx;
  mtx_init(&mtx, 0);
  if (mtx_timedlock(&mtx, 0) == thrd_success)
    mtx_unlock(&mtx);
  mtx_destroy(&mtx);
}
