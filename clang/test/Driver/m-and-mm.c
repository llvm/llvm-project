// RUN: %clang -M %s 2>&1 | FileCheck %s --implicit-check-not=warning
// RUN: %clang -MM %s 2>&1 | FileCheck %s --implicit-check-not=warning

// CHECK: m-and-mm.o:
// TEST-I: {{.*}}test.i:
// TEST: {{.*}}test:

// RUN: mkdir -p %t.dir

/// if -MD and -MMD are not specified, -o specifies the dependency file name.
// RUN: rm -f %t.dir/test.i
// RUN: %clang -M %s -o %t.dir/test.i
// RUN: FileCheck %s < %t.dir/test.i
// RUN: rm -f %t.dir/test.i
// RUN: %clang -MM %s -o %t.dir/test.i
// RUN: FileCheck %s < %t.dir/test.i

// RUN: rm -f %t.dir/test.d
// RUN: %clang -fsyntax-only -MD %s -o %t.dir/test.i
// RUN: FileCheck --check-prefix=TEST-I %s < %t.dir/test.d

// RUN: rm -f %t.dir/test.d
// RUN: %clang -M -MD %s -o %t.dir/test.i
// RUN: FileCheck --check-prefix=TEST-I %s < %t.dir/test.d

/// If the output file name does not have a suffix, just append `.d`.
// RUN: rm -f %t.dir/test.d
// RUN: %clang -fsyntax-only -MD %s -o %t.dir/test
// RUN: FileCheck --check-prefix=TEST %s < %t.dir/test.d

#warning "-M and -MM suppresses warnings, thus this warning shouldn't show up"
int main(void)
{
    return 0;
}
