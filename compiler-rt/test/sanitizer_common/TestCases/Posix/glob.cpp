// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <glob.h>
#include <string.h>

int main(void) {
  glob_t g;
  memset(&g, 0, sizeof(g));

  glob("*", 0, NULL, &g);
  globfree(&g);

  g.gl_offs = 1;
  glob("*", GLOB_DOOFFS, NULL, &g);
  assert(g.gl_pathv[0] == NULL);
  globfree(&g);

  return 0;
}
