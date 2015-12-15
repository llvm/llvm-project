// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

__attribute__((objc_root_class))
@interface Root
@end

@class Forward;

template <class T> void destroyPointer(T *t) {
  t->~T();
}

template <class T> void destroyReference(T &t) {
  t.~T();
}

template void destroyPointer<Root*>(Root **);
template void destroyReference<Root*>(Root *&);

// rdar://18522255
template void destroyPointer<Forward*>(Forward **);
template void destroyReference<Forward*>(Forward *&);
