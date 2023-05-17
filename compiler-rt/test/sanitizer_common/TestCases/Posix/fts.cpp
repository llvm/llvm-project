// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: darwin, target={{.*(linux|solaris).*}}

#include <sys/param.h>
#include <sys/types.h>

#include <sys/stat.h>

#include <assert.h>
#include <fts.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  char *const paths[] = {(char *)"/etc", 0};
  FTS *ftsp = fts_open(paths, FTS_LOGICAL, NULL);
  assert(ftsp);

  FTSENT *chp = fts_children(ftsp, 0);
  assert(chp);

  size_t n = 0;
  for (FTSENT *p = fts_read(ftsp); p; p = fts_read(ftsp)) {
    /* Skip recursively subdirectories */
    if (p->fts_info == FTS_D && p->fts_level != FTS_ROOTLEVEL) /* pre-order */
      fts_set(ftsp, p, FTS_SKIP);
    else if (p->fts_info == FTS_DP) /* post-order */
      continue;
    else if (p->fts_info == FTS_F) /* regular file */
      n++;
  }

  int rv = fts_close(ftsp);
  assert(!rv);

  printf("Number of files in /etc: '%zu'\n", n);

  return EXIT_SUCCESS;

  // CHECK: Number of files in /etc: '{{.*}}'
}
