// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: %clangxx -S -ftime-trace -ftime-trace-granularity=0 -o out %s
// RUN: %python %S/ftime-trace-sections.py < out.json

template <typename T>
void foo(T) {}
void bar() { foo(0); }
