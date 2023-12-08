// RUN: not %clang_cc1 -std=c++11 -E %s 2>&1 | grep 'error: raw string missing terminating delimiter )foo"'

const char *str = R"foo(abc
def)bar";
