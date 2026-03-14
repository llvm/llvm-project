int getBos1(void) {
  return __builtin_object_size(p, 0);
}

#define IS_CONST(x) __builtin_constant_p(x)
