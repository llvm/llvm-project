_Float16 try_builtin_fmaxf16(_Float16 x, _Float16 y) {
  return __builtin_fmaxf16(x, y);
}

_Float16 try_builtin_fminf16(_Float16 x, _Float16 y) {
  return __builtin_fminf16(x, y);
}

extern "C" void _start() {}
