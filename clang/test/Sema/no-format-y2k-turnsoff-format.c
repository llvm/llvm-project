// RUN: %clang_cc1 -verify -fsyntax-only -Wformat -Wno-format-y2k %s

void foo(const char *, ...) __attribute__((__format__ (__printf__, 1, 2)));

void bar(unsigned int a) {
        foo("%s", a); // expected-warning {{format specifies type 'char *' but the argument has type 'unsigned int'}}
}

