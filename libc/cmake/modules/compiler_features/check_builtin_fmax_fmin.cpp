_Float16 try_builtin_fmaxf16(_Float16 x, _Float16 y) {
  return __builtin_fmaxf16(x, y);
}
_Float16 try_builtin_fminf16(_Float16 x, _Float16 y) {
  return __builtin_fminf16(x, y);
}

float try_builtin_fmaxf(float x, float y) { return __builtin_fmaxf(x, y); }
float try_builtin_fminf(float x, float y) { return __builtin_fminf(x, y); }

double try_builtin_fmax(double x, double y) { return __builtin_fmax(x, y); }
double try_builtin_fmin(double x, double y) { return __builtin_fmin(x, y); }

extern "C" void _start() {}
