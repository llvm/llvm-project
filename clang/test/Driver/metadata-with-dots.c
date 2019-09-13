// REQUIRES: shell
// RUN: mkdir -p out.dir
// RUN: cat %s > out.dir/test.c
// RUN: %clang -E -MMD %s -o out.dir/test
// RUN: test ! -f %out.d
// RUN: test -f out.dir/test.d
// RUN: rm -rf out.dir/test.d out.dir/ out.d
int main (void)
{
    return 0;
}
