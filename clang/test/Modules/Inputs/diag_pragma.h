#define DIAG_PRAGMA_MACRO 1

#pragma clang diagnostic ignored "-Wparentheses"

#ifdef __cplusplus
template<typename T> const char *f(T t) {
  return "foo" + t;
}
#pragma clang diagnostic ignored "-Wstring-plus-int"
template<typename T> const char *g(T t) {
  return "foo" + t;
}
#endif
