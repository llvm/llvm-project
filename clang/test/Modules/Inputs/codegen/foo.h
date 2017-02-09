inline void f1(const char* fmt, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, fmt);
}
