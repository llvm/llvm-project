//--- No more than one '--config' may be specified.
//
// RUN: not %clang --config 1.cfg --config 2.cfg 2>&1 | FileCheck %s -check-prefix CHECK-DUPLICATE
// CHECK-DUPLICATE: no more than one option '--config' is allowed


//--- '--config' must be followed by config file name.
//
// RUN: not %clang --config 2>&1 | FileCheck %s -check-prefix CHECK-MISSING-FILE
// CHECK-MISSING-FILE: argument to '--config' is missing (expected 1 value)


//--- '--config' must not be found in config files.
//
// RUN: not %clang --config %S/Inputs/config-6.cfg 2>&1 | FileCheck %s -check-prefix CHECK-NESTED
// CHECK-NESTED: option '--config' is not allowed inside configuration file


//--- Argument of '--config' must be existing file, if it is specified by path.
//
// RUN: not %clang --config somewhere/nonexistent-config-file 2>&1 | FileCheck %s -check-prefix CHECK-NONEXISTENT
// CHECK-NONEXISTENT: configuration file '{{.*}}somewhere/nonexistent-config-file' does not exist


//--- Argument of '--config' must exist somewhere in well-known directories, if it is specified by bare name.
//
// RUN: not %clang --config-system-dir= --config-user-dir= --config nonexistent-config-file 2>&1 | FileCheck %s -check-prefix CHECK-NOTFOUND0
// CHECK-NOTFOUND0: configuration file 'nonexistent-config-file.cfg' cannot be found
// CHECK-NOTFOUND0-NEXT: was searched for in the directory:
// CHECK-NOTFOUND0-NOT: was searched for in the directory:
//
// RUN: not %clang --config-system-dir= --config-user-dir=%S/Inputs/config2 --config nonexistent-config-file 2>&1 | FileCheck %s -check-prefix CHECK-NOTFOUND1
// CHECK-NOTFOUND1: configuration file 'nonexistent-config-file.cfg' cannot be found
// CHECK-NOTFOUND1-NEXT: was searched for in the directory: {{.*}}/Inputs/config2
// CHECK-NOTFOUND1-NEXT: was searched for in the directory:
// CHECK-NOTFOUND1-NOT: was searched for in the directory:
//
// RUN: not %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config nonexistent-config-file 2>&1 | FileCheck %s -check-prefix CHECK-NOTFOUND2
// CHECK-NOTFOUND2: configuration file 'nonexistent-config-file.cfg' cannot be found
// CHECK-NOTFOUND2-NEXT: was searched for in the directory: {{.*}}/Inputs/config
// CHECK-NOTFOUND2-NEXT: was searched for in the directory:
// CHECK-NOTFOUND2-NOT: was searched for in the directory:
//
// RUN: not %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 --config nonexistent-config-file 2>&1 | FileCheck %s -check-prefix CHECK-NOTFOUND3
// CHECK-NOTFOUND3: configuration file 'nonexistent-config-file.cfg' cannot be found
// CHECK-NOTFOUND3-NEXT: was searched for in the directory: {{.*}}/Inputs/config2
// CHECK-NOTFOUND3-NEXT: was searched for in the directory: {{.*}}/Inputs/config
// CHECK-NOTFOUND3-NEXT: was searched for in the directory:


//--- Argument in config file cannot cross the file boundary
//
// RUN: not %clang --config %S/Inputs/config-5.cfg x86_64-unknown-linux-gnu -c %s 2>&1 | FileCheck %s -check-prefix CHECK-CROSS
// CHECK-CROSS: error: argument to '-target' is missing
