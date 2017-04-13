// RUN: %clang_analyze_cc1 -verify -analyzer-checker=core,alpha.security.ArrayBound -analyzer-store=region %s

struct tea_cheese { unsigned magic; };
typedef struct tea_cheese kernel_tea_cheese_t;
extern kernel_tea_cheese_t _wonky_gesticulate_cheese;

// This test case exercises the ElementRegion::getRValueType() logic.

void test1( void ) {
  kernel_tea_cheese_t *wonky = &_wonky_gesticulate_cheese;
  struct load_wine *cmd = (void*) &wonky[1];
  cmd = cmd;
  char *p = (void*) &wonky[1];
  kernel_tea_cheese_t *q = &wonky[1];
  // This test case tests both the RegionStore logic (doesn't crash) and
  // the out-of-bounds checking.  We don't expect the warning for now since
  // out-of-bound checking is temporarily disabled.
  kernel_tea_cheese_t r = *q; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void test1_b( void ) {
  kernel_tea_cheese_t *wonky = &_wonky_gesticulate_cheese;
  struct load_wine *cmd = (void*) &wonky[1];
  cmd = cmd;
  char *p = (void*) &wonky[1];
  *p = 1;  // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}
