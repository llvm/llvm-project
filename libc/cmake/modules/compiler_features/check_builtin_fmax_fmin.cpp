float try_builtin_fmaxf(float x, float y) { return __builtin_fmaxf(x, y); }
float try_builtin_fminf(float x, float y) { return __builtin_fminf(x, y); }

double try_builtin_fmaxf(double x, double y) { return __builtin_fmax(x, y); }
double try_builtin_fminf(double x, double y) { return __builtin_fmin(x, y); }

extern "C" void _start() {}
