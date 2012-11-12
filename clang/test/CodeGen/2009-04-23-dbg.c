// RUN: %clang_cc1 -S -g -o %t %s
# 1 "a.c"
# 1 "a.c" 1
# 1 "<built-in>" 1
# 103 "<built-in>"
# 103 "<command line>" 1

# 1 "/private/tmp/a.h" 1
int bar;
# 105 "<command line>" 2
# 105 "<built-in>" 2
# 1 "a.c" 2
# 1 "/private/tmp/a.h" 1
int bar;
# 2 "a.c" 2

int main() {
 bar = 0;
 return 0;
}
