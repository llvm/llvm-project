// RUN: %clang -M -MM %s 2>&1 | FileCheck /dev/null --implicit-check-not=warning

// RUN: mkdir -p %t.dir
// RUN: rm -f %t.dir/test.d
// RUN: %clang -fsyntax-only -MD %s -o %t.dir/test.i
// RUN: test -f %t.dir/test.d

/// If the output file name does not have a suffix, just append `.d`.
// RUN: rm -f %t.dir/test.d
// RUN: %clang -fsyntax-only -MD %s -o %t.dir/test
// RUN: test -f %t.dir/test.d

#warning "-M and -MM suppresses preprocessing, thus this warning shouldn't show up"
int main(void)
{
    return 0;
}
