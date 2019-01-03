// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, solaris, darwin

#include <ctype.h>
#include <err.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vis.h>

void test_vis() {
  char visout[5];
  int ch = toascii(0x1);
  vis(visout, ch, VIS_SAFE | VIS_NOSLASH, 0);
  printf("vis: %s\n", visout);
}

void test_nvis() {
  char visout[5];
  int ch = toascii(0x2);
  nvis(visout, sizeof visout, ch, VIS_SAFE | VIS_NOSLASH, 0);
  printf("nvis: %s\n", visout);
}

void test_strvis() {
  char visout[5];
  strvis(visout, "\3", VIS_SAFE | VIS_NOSLASH);
  printf("strvis: %s\n", visout);
}

void test_stravis() {
  char *visout;
  stravis(&visout, "\4", VIS_SAFE | VIS_NOSLASH);
  printf("stravis: %s\n", visout);
  free(visout);
}

void test_strnvis() {
  char visout[5];
  strnvis(visout, sizeof visout, "\5", VIS_SAFE | VIS_NOSLASH);
  printf("strnvis: %s\n", visout);
}

void test_strvisx() {
  char visout[5];
  char src[] = "\6";
  strvisx(visout, src, sizeof src - 1 /* skip final \0 */,
          VIS_SAFE | VIS_NOSLASH);
  printf("strvisx: %s\n", visout);
}

void test_strnvisx() {
  char visout[5];
  char src[] = "\1";
  strnvisx(visout, sizeof visout, src, sizeof src - 1 /* skip final \0 */,
           VIS_SAFE | VIS_NOSLASH);
  printf("strnvisx: %s\n", visout);
}

void test_strenvisx() {
  char visout[5];
  char src[] = "\2";
  strenvisx(visout, sizeof visout, src, sizeof src - 1 /* skip final \0 */,
            VIS_SAFE | VIS_NOSLASH, NULL);
  printf("strenvisx: %s\n", visout);
}

void test_svis() {
  char visout[5];
  int ch = toascii(0x3);
  svis(visout, ch, VIS_SAFE | VIS_NOSLASH, 0, "x");
  printf("svis: %s\n", visout);
}

void test_snvis() {
  char visout[5];
  int ch = toascii(0x2);
  snvis(visout, sizeof visout, ch, VIS_SAFE | VIS_NOSLASH, 0, "x");
  printf("snvis: %s\n", visout);
}

void test_strsvis() {
  char visout[5];
  strsvis(visout, "\4", VIS_SAFE | VIS_NOSLASH, "x");
  printf("strsvis: %s\n", visout);
}

void test_strsnvis() {
  char visout[5];
  strsnvis(visout, sizeof visout, "\5", VIS_SAFE | VIS_NOSLASH, "x");
  printf("strsnvis: %s\n", visout);
}

void test_strsvisx() {
  char visout[5];
  char src[] = "\5";
  strsvisx(visout, src, sizeof src - 1 /* skip final \0 */,
           VIS_SAFE | VIS_NOSLASH, "x");
  printf("strsvisx: %s\n", visout);
}

void test_strsnvisx() {
  char visout[5];
  char src[] = "\6";
  strsnvisx(visout, sizeof visout, src, sizeof src - 1 /* skip final \0 */,
            VIS_SAFE | VIS_NOSLASH, "x");
  printf("strsnvisx: %s\n", visout);
}

void test_strsenvisx() {
  char visout[5];
  char src[] = "\1";
  strsenvisx(visout, sizeof visout, src, sizeof src - 1 /* skip final \0 */,
             VIS_SAFE | VIS_NOSLASH, "x", NULL);
  printf("strsenvisx: %s\n", visout);
}

void test_unvis() {
  char visout[5];
  int ch = toascii(0x1);
  vis(visout, ch, VIS_SAFE, 0);

  int state = 0;
  char out;
  char *p = visout;
  while ((ch = *(p++)) != '\0') {
  again:
    switch (unvis(&out, ch, &state, 0)) {
    case 0:
    case UNVIS_NOCHAR:
      break;
    case UNVIS_VALID:
      printf("unvis: %" PRIx8 "\n", (unsigned char)out);
      break;
    case UNVIS_VALIDPUSH:
      printf("unvis: %" PRIx8 "\n", (unsigned char)out);
      goto again;
    case UNVIS_SYNBAD:
      errx(1, "Bad character sequence!");
    }
  }
  if (unvis(&out, '\0', &state, UNVIS_END) == UNVIS_VALID)
    printf("unvis: %" PRIx8 "\n", (unsigned char)out);
}

void test_strunvis() {
  char visout[5];
  int ch = toascii(0x2);
  vis(visout, ch, VIS_SAFE, 0);

  char p[5];
  strunvis(p, visout);

  char *pp = p;
  while ((ch = *(pp++)) != '\0')
    printf("strunvis: %" PRIx8 "\n", (unsigned char)ch);
}

void test_strnunvis() {
  char visout[5];
  int ch = toascii(0x3);
  vis(visout, ch, VIS_SAFE, 0);

  char p[5];
  strnunvis(p, sizeof p, visout);

  char *pp = p;
  while ((ch = *(pp++)) != '\0')
    printf("strnunvis: %" PRIx8 "\n", (unsigned char)ch);
}

void test_strunvisx() {
  char visout[5];
  int ch = toascii(0x4);
  vis(visout, ch, VIS_SAFE, 0);

  char p[5];
  strunvisx(p, visout, VIS_SAFE);

  char *pp = p;
  while ((ch = *(pp++)) != '\0')
    printf("strunvisx: %" PRIx8 "\n", (unsigned char)ch);
}

void test_strnunvisx() {
  char visout[5];
  int ch = toascii(0x5);
  vis(visout, ch, VIS_SAFE, 0);

  char p[5];
  strnunvisx(p, sizeof p, visout, VIS_SAFE);

  char *pp = p;
  while ((ch = *(pp++)) != '\0')
    printf("strnunvisx: %" PRIx8 "\n", (unsigned char)ch);
}

int main(void) {
  printf("vis\n");

  test_vis();
  test_nvis();
  test_strvis();
  test_stravis();
  test_strnvis();
  test_strvisx();
  test_strnvisx();
  test_strenvisx();
  test_svis();
  test_snvis();
  test_strsvis();
  test_strsnvis();
  test_strsvisx();
  test_strsnvisx();
  test_strsenvisx();
  test_unvis();
  test_strunvis();
  test_strnunvis();
  test_strunvisx();
  test_strnunvisx();

  // CHECK: vis
  // CHECK: vis: ^A
  // CHECK: nvis: ^B
  // CHECK: strvis: ^C
  // CHECK: stravis: ^D
  // CHECK: strnvis: ^E
  // CHECK: strvisx: ^F
  // CHECK: strnvisx: ^A
  // CHECK: strenvisx: ^B
  // CHECK: svis: ^C
  // CHECK: snvis: ^B
  // CHECK: strsvis: ^D
  // CHECK: strsnvis: ^E
  // CHECK: strsvisx: ^E
  // CHECK: strsnvisx: ^F
  // CHECK: strsenvisx: ^A
  // CHECK: unvis: 1
  // CHECK: strunvis: 2
  // CHECK: strnunvis: 3
  // CHECK: strunvisx: 4
  // CHECK: strnunvisx: 5

  return 0;
}
