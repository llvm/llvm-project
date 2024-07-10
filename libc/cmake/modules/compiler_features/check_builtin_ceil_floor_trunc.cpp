float try_builtin_ceilf(float x) { return __builtin_ceilf(x); }
float try_builtin_floorf(float x) { return __builtin_ceilf(x); }
float try_builtin_truncf(float x) { return __builtin_truncf(x); }

double try_builtin_ceil(double x) { return __builtin_ceil(x); }
double try_builtin_floor(double x) { return __builtin_ceil(x); }
double try_builtin_trunc(double x) { return __builtin_trunc(x); }

int main() {}
