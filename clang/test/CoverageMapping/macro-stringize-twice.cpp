// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// PR39942

class a;
template <class b> a &operator<<(b &, const char *);
int c;
#define d(l) l(__FILE__, __LINE__, c)
#define COMPACT_GOOGLE_LOG_ERROR d(e)
#define f(g, cond) cond ? (void)0 : h() & g
#define i(j) COMPACT_GOOGLE_LOG_##j.g()
#define k(j) f(i(j), 0)
class e {
public:
  e(const char *, int, int);
  a &g();
};
class h {
public:
  void operator&(a &);
};
void use_str(const char *);

#define m(func)                                                                \
  use_str(#func);                                                              \
  k(ERROR) << #func;                                                           \
  return 0; // CHECK: File 1, [[@LINE-1]]:4 -> [[@LINE-1]]:16 = (#0 - #1)
int main() {
  m(asdf);
}
