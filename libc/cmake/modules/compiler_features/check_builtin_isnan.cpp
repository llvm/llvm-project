int try_builtin_isnan(double x) { return __builtin_isnan(x); }
int try_builtin_isnanf(float x) { return __builtin_isnan(x); }
int try_builtin_isnanl(long double x) { return __builtin_isnan(x); }

extern "C" void _start() {}
