// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// RUN: %clangxx_asan -O1 %s -o %t && %run %t 2>&1
// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=250
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <assert.h>

int main() {
  int id = shmget(IPC_PRIVATE, 4096, 0644 | IPC_CREAT);
  assert(id > -1);
  struct shmid_ds ds;
  int res = shmctl(id, IPC_STAT, &ds);
  assert(res > -1);
  printf("shm_segsz: %zd\n", ds.shm_segsz);
  assert(ds.shm_segsz == 4096);
  assert(-1 != shmctl(id, IPC_RMID, 0));

  struct shm_info shmInfo;
  res = shmctl(0, SHM_INFO, (struct shmid_ds *)&shmInfo);
  assert(res > -1);
  
  return 0;
}
