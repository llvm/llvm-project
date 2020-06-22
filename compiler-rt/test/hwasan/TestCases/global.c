// RUN: %clang_hwasan %s -o %t
// RUN: %run %t 0
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RSYM %s
// RUN: not %env_hwasan_opts=symbolize=0 %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RNOSYM %s
// RUN: not %run %t -1 2>&1 | FileCheck --check-prefixes=CHECK,LSYM %s
// RUN: not %env_hwasan_opts=symbolize=0 %run %t -1 2>&1 | FileCheck --check-prefixes=CHECK,LNOSYM %s

int x = 1;

int main(int argc, char **argv) {
  // RSYM: is located 0 bytes to the right of 4-byte global variable x {{.*}} in {{.*}}global.c.tmp
  // RNOSYM: is located to the right of a 4-byte global variable in ({{.*}}global.c.tmp+{{.*}})
  // LSYM: is located 4 bytes to the left of 4-byte global variable x {{.*}} in {{.*}}global.c.tmp
  // LNOSYM: is located to the left of a 4-byte global variable in ({{.*}}global.c.tmp+{{.*}})
  // CHECK-NOT: can not describe
  (&x)[atoi(argv[1])] = 1;
}
