// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

struct Word {
  short bar;
  short baz;
  int final:1;
  short quux;
} *word_limit;

void foo ()
{
  word_limit->final = (word_limit->final && word_limit->final);
}
