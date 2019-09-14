// REQUIRES: shell
// RUN: mkdir -p %t/out.dir
// RUN: cat %s > %t/out.dir/test.c
// RUN: %clang -E -MMD %s -o %t/out.dir/test
// RUN: test ! -f %out.d
// RUN: test -f %t/out.dir/test.d
// RUN: rm -rf %t/out.dir/test.d %t/out.dir/ out.d
int main (void)
{
    return 0;
}
