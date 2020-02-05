// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.unix.PthreadLock -verify %s

// Tests performing normal locking patterns and wrong locking orders

#include "Inputs/system-header-simulator-for-pthread-lock.h"

pthread_mutex_t mtx1, mtx2;
pthread_mutex_t *pmtx;
lck_mtx_t lck1, lck2;
lck_grp_t grp1;
lck_rw_t rw;

#define NULL 0

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
ok8(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
}

void
ok9(void)
{
	pthread_mutex_unlock(&mtx1);		// no-warning
	if (pthread_mutex_trylock(&mtx1) == 0)	// no-warning
		pthread_mutex_unlock(&mtx1);	// no-warning
}

void
ok10(void)
{
	if (pthread_mutex_trylock(&mtx1) != 0)	// no-warning
		pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);		// no-warning
}

void
ok11(void)
{
	pthread_mutex_destroy(&mtx1);	// no-warning
}

void
ok12(void)
{
	pthread_mutex_destroy(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx2);	// no-warning
}

void
ok13(void)
{
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx1);	// no-warning
}

void
ok14(void)
{
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
	pthread_mutex_destroy(&mtx2);	// no-warning
}

void
ok15(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx1);	// no-warning
}

void
ok16(void)
{
	pthread_mutex_init(&mtx1, NULL);	// no-warning
}

void
ok17(void)
{
	pthread_mutex_init(&mtx1, NULL);	// no-warning
	pthread_mutex_init(&mtx2, NULL);	// no-warning
}

void
ok18(void)
{
	pthread_mutex_destroy(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// no-warning
}

void
ok19(void)
{
	pthread_mutex_destroy(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// no-warning
	pthread_mutex_destroy(&mtx2);		// no-warning
	pthread_mutex_init(&mtx2, NULL);	// no-warning
}

void
ok20(void)
{
	pthread_mutex_unlock(&mtx1);		// no-warning
	pthread_mutex_destroy(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// no-warning
	pthread_mutex_destroy(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// no-warning
}

void
ok21(void) {
  pthread_mutex_lock(pmtx);    // no-warning
  pthread_mutex_unlock(pmtx);  // no-warning
}

void
ok22(void) {
  pthread_mutex_lock(pmtx);    // no-warning
  pthread_mutex_unlock(pmtx);  // no-warning
  pthread_mutex_lock(pmtx);    // no-warning
  pthread_mutex_unlock(pmtx);  // no-warning
}

void ok23(void) {
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_destroy(&mtx1);        // no-warning
}

void ok24(void) {
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_lock(&mtx1);           // no-warning
}

void ok25(void) {
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_unlock(&mtx1);         // no-warning
}

void ok26(void) {
  pthread_mutex_unlock(&mtx1);           // no-warning
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_lock(&mtx1);           // no-warning
}

void ok27(void) {
  pthread_mutex_unlock(&mtx1);           // no-warning
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_lock(&mtx1);           // no-warning
  else
    pthread_mutex_init(&mtx1, NULL); // no-warning
}

void ok28(void) {
  if (pthread_mutex_destroy(&mtx1) != 0) { // no-warning
    pthread_mutex_lock(&mtx1);             // no-warning
    pthread_mutex_unlock(&mtx1);           // no-warning
    pthread_mutex_destroy(&mtx1);          // no-warning
  }
}

void ok29(void) {
  lck_rw_lock_shared(&rw);
  lck_rw_unlock_shared(&rw);
  lck_rw_lock_exclusive(&rw); // no-warning
  lck_rw_unlock_exclusive(&rw); // no-warning
}

void escape_mutex(pthread_mutex_t *m);
void ok30(void) {
  pthread_mutex_t local_mtx;
  pthread_mutex_init(&local_mtx, NULL);
  pthread_mutex_lock(&local_mtx);
  escape_mutex(&local_mtx);
  pthread_mutex_lock(&local_mtx); // no-warning
  pthread_mutex_unlock(&local_mtx);
  pthread_mutex_destroy(&local_mtx);
}

void ok31(void) {
  pthread_mutex_t local_mtx;
  pthread_mutex_init(&local_mtx, NULL);
  pthread_mutex_lock(&local_mtx);
  fake_system_function_that_takes_a_mutex(&local_mtx);
  pthread_mutex_lock(&local_mtx); // no-warning
  pthread_mutex_unlock(&local_mtx);
  pthread_mutex_destroy(&local_mtx);
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

void
bad9(void)
{
	lck_mtx_unlock(&lck1);		// no-warning
	lck_mtx_unlock(&lck1);		// expected-warning{{This lock has already been unlocked}}
}

void
bad10(void)
{
	lck_mtx_lock(&lck1);		// no-warning
	lck_mtx_unlock(&lck1);		// no-warning
	lck_mtx_unlock(&lck1);		// expected-warning{{This lock has already been unlocked}}
}

static void
bad11_sub(pthread_mutex_t *lock)
{
	lck_mtx_unlock(lock);		// expected-warning{{This lock has already been unlocked}}
}

void
bad11(int i)
{
	lck_mtx_lock(&lck1);		// no-warning
	lck_mtx_unlock(&lck1);		// no-warning
	if (i < 5)
		bad11_sub(&lck1);
}

void
bad12(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// expected-warning{{This lock has already been unlocked}}
}

void
bad13(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// expected-warning{{This lock has already been unlocked}}
}

void
bad14(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx2);	// expected-warning{{This lock has already been unlocked}}
}

void
bad15(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx2);	// no-warning
	pthread_mutex_unlock(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx2);	// expected-warning{{This lock has already been unlocked}}
}

void
bad16(void)
{
	pthread_mutex_destroy(&mtx1);	// no-warning
	pthread_mutex_lock(&mtx1);	// expected-warning{{This lock has already been destroyed}}
}

void
bad17(void)
{
	pthread_mutex_destroy(&mtx1);	// no-warning
	pthread_mutex_unlock(&mtx1);	// expected-warning{{This lock has already been destroyed}}
}

void
bad18(void)
{
	pthread_mutex_destroy(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx1);	// expected-warning{{This lock has already been destroyed}}
}

void
bad19(void)
{
	pthread_mutex_lock(&mtx1);	// no-warning
	pthread_mutex_destroy(&mtx1);	// expected-warning{{This lock is still locked}}
}

void
bad20(void)
{
	lck_mtx_destroy(&mtx1, &grp1);	// no-warning
	lck_mtx_lock(&mtx1);		// expected-warning{{This lock has already been destroyed}}
}

void
bad21(void)
{
	lck_mtx_destroy(&mtx1, &grp1);	// no-warning
	lck_mtx_unlock(&mtx1);		// expected-warning{{This lock has already been destroyed}}
}

void
bad22(void)
{
	lck_mtx_destroy(&mtx1, &grp1);	// no-warning
	lck_mtx_destroy(&mtx1, &grp1);	// expected-warning{{This lock has already been destroyed}}
}

void
bad23(void)
{
	lck_mtx_lock(&mtx1);		// no-warning
	lck_mtx_destroy(&mtx1, &grp1);	// expected-warning{{This lock is still locked}}
}

void
bad24(void)
{
	pthread_mutex_init(&mtx1, NULL);	// no-warning
	pthread_mutex_init(&mtx1, NULL);	// expected-warning{{This lock has already been initialized}}
}

void
bad25(void)
{
	pthread_mutex_lock(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// expected-warning{{This lock is still being held}}
}

void
bad26(void)
{
	pthread_mutex_unlock(&mtx1);		// no-warning
	pthread_mutex_init(&mtx1, NULL);	// expected-warning{{This lock has already been initialized}}
}

void bad27(void) {
  pthread_mutex_unlock(&mtx1);            // no-warning
  int ret = pthread_mutex_destroy(&mtx1); // no-warning
  if (ret != 0)                           // no-warning
    pthread_mutex_lock(&mtx1);            // no-warning
  else
    pthread_mutex_unlock(&mtx1); // expected-warning{{This lock has already been destroyed}}
}

void bad28(void) {
  pthread_mutex_unlock(&mtx1);            // no-warning
  int ret = pthread_mutex_destroy(&mtx1); // no-warning
  if (ret != 0)                           // no-warning
    pthread_mutex_lock(&mtx1);            // no-warning
  else
    pthread_mutex_lock(&mtx1); // expected-warning{{This lock has already been destroyed}}
}

void bad29(void) {
  pthread_mutex_lock(&mtx1);             // no-warning
  pthread_mutex_unlock(&mtx1);           // no-warning
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_init(&mtx1, NULL);     // expected-warning{{This lock has already been initialized}}
  else
    pthread_mutex_init(&mtx1, NULL); // no-warning
}

void bad30(void) {
  pthread_mutex_lock(&mtx1);             // no-warning
  pthread_mutex_unlock(&mtx1);           // no-warning
  if (pthread_mutex_destroy(&mtx1) != 0) // no-warning
    pthread_mutex_init(&mtx1, NULL);     // expected-warning{{This lock has already been initialized}}
  else
    pthread_mutex_destroy(&mtx1); // expected-warning{{This lock has already been destroyed}}
}

void bad31(void) {
  int ret = pthread_mutex_destroy(&mtx1); // no-warning
  pthread_mutex_lock(&mtx1);              // expected-warning{{This lock has already been destroyed}}
  if (ret != 0)
    pthread_mutex_lock(&mtx1);
}

void bad32(void) {
  lck_rw_lock_shared(&rw);
  lck_rw_unlock_exclusive(&rw); // FIXME: warn - should be shared?
  lck_rw_lock_exclusive(&rw);
  lck_rw_unlock_shared(&rw); // FIXME: warn - should be exclusive?
}

void bad33(void) {
  pthread_mutex_lock(pmtx);
  fake_system_function();
  pthread_mutex_lock(pmtx); // expected-warning{{This lock has already been acquired}}
}
