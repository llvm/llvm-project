#ifndef TEST__MAKE_SIGNED_H
#define TEST__MAKE_SIGNED_H

static int my_make_signed(unsigned a) {
  // Use different return paths so each report location can be distinguished.
  if (a < 4002222222U)
    return a;
  if (a < 4003333333U)
    return a;
  return a;
}

#endif
