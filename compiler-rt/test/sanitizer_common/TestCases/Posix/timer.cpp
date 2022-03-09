// RUN: %clangxx -O0 -g %s -o %t && %run %t
// UNSUPPORTED: darwin
// FIXME: SEGV - API mismatch?
// UNSUPPORTED: s390
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
  struct sigevent sev {};
  sev.sigev_notify = SIGEV_NONE;
  timer_t timerid;
  assert(timer_create(CLOCK_REALTIME, &sev, &timerid) == 0);

  struct itimerspec new_value {};
  new_value.it_value.tv_sec = 10;
  new_value.it_value.tv_nsec = 1000000;
  new_value.it_interval.tv_sec = new_value.it_value.tv_sec;
  new_value.it_interval.tv_nsec = new_value.it_value.tv_nsec;

  assert(timer_settime(timerid, 0, &new_value, nullptr) == 0);

  struct itimerspec old_value;
  assert(timer_settime(timerid, 0, &new_value, &old_value) == 0);
  assert(old_value.it_interval.tv_sec == new_value.it_interval.tv_sec);
  assert(old_value.it_interval.tv_nsec == new_value.it_interval.tv_nsec);
  assert(old_value.it_value.tv_sec <= new_value.it_value.tv_sec);
  assert(old_value.it_value.tv_nsec <= new_value.it_value.tv_nsec);

  struct itimerspec curr_value;
  assert(timer_gettime(timerid, &curr_value) == 0);
  assert(curr_value.it_interval.tv_sec == new_value.it_interval.tv_sec);
  assert(curr_value.it_interval.tv_nsec == new_value.it_interval.tv_nsec);
  assert(curr_value.it_value.tv_sec <= new_value.it_value.tv_sec);
  assert(curr_value.it_value.tv_nsec <= new_value.it_value.tv_nsec);

  assert(timer_delete(timerid) == 0);

  return 0;
}
