@import top;

template<typename T>
void f(int k, ...) {
  va_list va;
  __builtin_va_start(va, k);
}
