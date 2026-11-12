#define STATIC_CAST static_cast

template<typename T>
void foo(T *p) {
  (void)STATIC_CAST<T*>(0);
}
