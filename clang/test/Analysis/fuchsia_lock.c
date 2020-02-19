// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.fuchsia.Lock -verify %s

typedef int spin_lock_t;
typedef int zx_status_t;
typedef int zx_time_t;

void spin_lock(spin_lock_t *lock);
int spin_trylock(spin_lock_t *lock);
void spin_unlock(spin_lock_t *lock);
void spin_lock_init(spin_lock_t *lock);

void spin_lock_save(spin_lock_t *lock, void *statep,
                    int flags);
void spin_unlock_restore(spin_lock_t *lock, void *old_state,
                         int flags);

spin_lock_t mtx1;
spin_lock_t mtx2;

void bad1(void)
{
	spin_lock(&mtx1);	// no-warning
	spin_lock(&mtx1);	// expected-warning{{This lock has already been acquired}}
}

void bad2(void) {
  spin_lock(&mtx1);
  spin_unlock(&mtx1);
  spin_unlock(&mtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad3() {
  spin_lock_init(&mtx1);
  if (spin_trylock(&mtx1) != 0)
    spin_unlock(&mtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad4(void) {
  spin_lock(&mtx1);
  spin_lock(&mtx2);
  spin_unlock(&mtx1); // expected-warning {{This was not the most recently acquired lock. Possible lock order reversal}}
  spin_unlock(&mtx2);
}

void good() {
  spin_lock_t mtx;
  spin_lock_init(&mtx);
  spin_lock_save(&mtx, 0, 0);
  spin_unlock_restore(&mtx, 0, 0);
}

void good2() {
  spin_lock_t mtx;
  spin_lock_init(&mtx);
  if (spin_trylock(&mtx) == 0)
    spin_unlock(&mtx);
}

typedef int sync_mutex_t;
void sync_mutex_lock(sync_mutex_t* mutex);
void sync_mutex_lock_with_waiter(sync_mutex_t* mutex);
zx_status_t sync_mutex_timedlock(sync_mutex_t* mutex, zx_time_t deadline);
zx_status_t sync_mutex_trylock(sync_mutex_t* mutex);
void sync_mutex_unlock(sync_mutex_t* mutex);

sync_mutex_t smtx1;
sync_mutex_t smtx2;

void bad11(void)
{
	sync_mutex_lock(&smtx1);	// no-warning
	sync_mutex_lock(&smtx1);	// expected-warning{{This lock has already been acquired}}
}

void bad12(void) {
  sync_mutex_lock_with_waiter(&smtx1);
  sync_mutex_unlock(&smtx1);
  sync_mutex_unlock(&smtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad13() {
  sync_mutex_unlock(&smtx1);
  if (sync_mutex_trylock(&smtx1) != 0)
    sync_mutex_unlock(&smtx1); // expected-warning {{This lock has already been unlocked}}
}

void bad14(void) {
  sync_mutex_lock(&smtx1);
  sync_mutex_lock(&smtx2);
  sync_mutex_unlock(&smtx1); // expected-warning {{This was not the most recently acquired lock. Possible lock order reversal}}
  sync_mutex_unlock(&smtx2);
}

void good11() {
  sync_mutex_t mtx;
  if (sync_mutex_trylock(&mtx) == 0)
    sync_mutex_unlock(&mtx);
}

void good12() {
  sync_mutex_t mtx;
  if (sync_mutex_timedlock(&mtx, 0) == 0)
    sync_mutex_unlock(&mtx);
}
