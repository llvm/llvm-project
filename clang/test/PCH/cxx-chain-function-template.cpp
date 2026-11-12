// RUN: %clang_cc1 -chain-include %s -chain-include %s -fsyntax-only %s
// Just don't crash.
#if !defined(RUN1)
#define RUN1

struct CXXRecordDecl { CXXRecordDecl(int); };

template <typename T, typename U>
T cast(U u) {
  return reinterpret_cast<T&>(u);
}

void test1() {
  cast<float>(1);
}

#elif !defined(RUN2)
#define RUN2

template <typename T>
void test2(T) {
  cast<CXXRecordDecl>(1.0f);
}

#else

void test3() {
  cast<CXXRecordDecl>(1.0f);
  test2(1);
}

#endif
