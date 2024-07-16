_Float16 try_builtin_frexpf16(_Float16 x, int *exp) {
  return __builtin_frexpf16(x, exp);
}

extern "C" void _start() {}
