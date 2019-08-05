// RUN: %clangxx -O0 -g %s -o %t && %run %t | FileCheck %s
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/timerfd.h>
#include <unistd.h>

int main (int argc, char** argv)
{
  int fd = timerfd_create(CLOCK_REALTIME, 0);
  assert(fd >= 0);

  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = 1000000;
  its.it_interval.tv_sec = its.it_value.tv_sec;
  its.it_interval.tv_nsec = its.it_value.tv_nsec;

  int res = timerfd_settime(fd, 0, &its, NULL);
  assert(res != -1);

  struct itimerspec its2;
  res = timerfd_settime(fd, 0, &its, &its2);
  assert(res != -1);
  assert(its2.it_interval.tv_sec == its.it_interval.tv_sec);
  assert(its2.it_interval.tv_nsec == its.it_interval.tv_nsec);
  assert(its2.it_value.tv_sec <= its.it_value.tv_sec);
  assert(its2.it_value.tv_nsec <= its.it_value.tv_nsec);

  struct itimerspec its3;
  res = timerfd_gettime(fd, &its3);
  assert(res != -1);
  assert(its3.it_interval.tv_sec == its.it_interval.tv_sec);
  assert(its3.it_interval.tv_nsec == its.it_interval.tv_nsec);
  assert(its3.it_value.tv_sec <= its.it_value.tv_sec);
  assert(its3.it_value.tv_nsec <= its.it_value.tv_nsec);


  unsigned long long buf;
  res = read(fd, &buf, sizeof(buf));
  assert(res == 8);
  assert(buf >= 1);

  res = close(fd);
  assert(res != -1);

  printf("DONE\n");
  // CHECK: DONE
  
  return 0;
}
