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
// CHECK-NEXT: WRITE CALLED: test
// CHECK-NEXT: FLUSH CALLED
// CHECK-NEXT: WRITE CALLED: test
// CHECK-NEXT: FLUSH CALLED

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cookie_var;

ssize_t f_read(void *cookie, void *buf, size_t len) {
  assert(cookie == &cookie_var);
  assert(len >= 6);
  printf("READ CALLED; len=%zd\n", len);
  return strlcpy((char*)buf, "test\n", len);
}

ssize_t f_write(void *cookie, const void *buf, size_t len) {
  assert(cookie == &cookie_var);
  char *data = strndup((char*)buf, len);
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

int f_flush(void *cookie) {
  assert(cookie == &cookie_var);
  printf("FLUSH CALLED\n");
  return 0;
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
  fp = fropen2(&cookie_var, f_read);
  assert(fp);
  // verify that fileno() does not crash or report nonsense
  assert(fileno(fp) == -1);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(!fclose(fp));

  // 2. write-only variant
  fp = fwopen2(&cookie_var, f_write);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fputs("test", fp) >= 0);
  assert(!fclose(fp));

  // 3. read+write+close
  fp = funopen2(&cookie_var, f_read, f_write, NULL, NULL, f_close);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(fputs("test", fp) >= 0);
  assert(!fflush(fp));
  assert(!fclose(fp));

  // 4. read+seek
  fp = funopen2(&cookie_var, f_read, NULL, f_seek, NULL, NULL);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fseek(fp, 100, SEEK_SET) == 0);
  assert(fgets(buf, sizeof(buf), fp));
  printf("READ: %s", buf);
  assert(!fclose(fp));

  // 5. write+flush
  fp = funopen2(&cookie_var, NULL, f_write, NULL, f_flush, NULL);
  assert(fp);
  assert(fileno(fp) == -1);
  assert(fputs("test", fp) >= 0);
  assert(!fflush(fp));
  assert(fputs("test", fp) >= 0);
  // NB: fclose() also implicitly calls flush
  assert(!fclose(fp));

  return 0;
}
