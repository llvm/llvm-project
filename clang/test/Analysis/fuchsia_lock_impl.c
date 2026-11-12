// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.fuchsia.Lock -verify %s
// expected-no-diagnostics
typedef int spin_lock_t;

void spin_lock(spin_lock_t *lock);
int getCond(void);
int spin_trylock(spin_lock_t *lock) {
    if (getCond())
        return 0;
    return -1;
}
void spin_unlock(spin_lock_t *lock);

spin_lock_t mtx;
void no_crash(void) {
  if (spin_trylock(&mtx) == 0)
    spin_unlock(&mtx);
}
