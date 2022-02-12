// RUN: %clang_cc1 -E -std=c++11 %s -o %t
// RUN: %clang_cc1 %t

#define FOO(str) foo(#str)

extern void foo(const char *str);

void bar(void) {
  FOO(R"(foo
    bar)");
}
