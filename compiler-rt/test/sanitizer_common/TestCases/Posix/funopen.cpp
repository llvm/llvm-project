// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s

// CHECK: READ CALLED; len={{[0-9]*}}
// CHECK-NEXT: READ: test
// CHECK-NEXT: WRITE CALLED: test
// CHECK-NEXT: READ CALLED; len={{[0-9]*}}
// CHECK-NEXT: READ: test
// CHECK-NEXT: WRITE CALLED: test
// CHECK-NEXT: CLOSE CALLED
// CHECK-NEXT: SEEK CALLED; off=100, whence=0
// CHECK-NEXT: READ CALLED; len={{[0-9]*}}
// CHECK-NEXT: READ: test
//
// UNSUPPORTED: darwin, target={{.*(linux|solaris).*}}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cookie_var;

int f_read(void *cookie, char *buf, int len) {
  assert(cookie == &cookie_var);
  assert(len >= 6);
  printf("READ CALLED; len=%d\n", len);
  return strlcpy(buf, "test\n", len);
}

int f_write(void *cookie, const char *buf, int len) {
  assert(cookie == &cookie_var);
  char *data = strndup(buf, len);
  assert(data);
  printf("WRITE CALLED: %s\n", data);
  free(data);
  return len;
}

off_t f_seek(void *cookie, off_t off, int whence) {
  assert(cookie == &cookie_var);
  assert(whence == SEEK_SET);
  printf("SEEK CALLED; off=%d, whence=%d\n", (int)off, whence);
  return off;
}

int f_close(void *cookie) {
  assert(cookie == &cookie_var);
  printf("CLOSE CALLED\n");
  return 0;
}

int main(void) {
  FILE *fp;
  char buf[10];

  // 1. read-only variant
  fp = fropen(&cookie_var, f_read);
  assert(fp);
  // verify that fileno() does not crash or report nonsense
  assert(fileno(fp) == -1);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(!fclose(fp));

  // 2. write-only variant
  fp = fwopen(&cookie_var, f_write);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fputs("test", fp) >= 0);
  assert(!fclose(fp));

  // 3. read+write+close
  fp = funopen(&cookie_var, f_read, f_write, NULL, f_close);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(fputs("test", fp) >= 0);
  assert(!fclose(fp));

  // 4. read+seek
  fp = funopen(&cookie_var, f_read, NULL, f_seek, NULL);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fseek(fp, 100, SEEK_SET) == 0);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(!fclose(fp));

  return 0;
}
