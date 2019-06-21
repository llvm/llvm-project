// REQUIRES: lld

// RUN: %clang -g -c -o %t-1.o --target=x86_64-pc-linux -mllvm -accel-tables=Disable %s
// RUN: %clang -g -c -o %t-2.o --target=x86_64-pc-linux -mllvm -accel-tables=Disable %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// Run the same test with split-dwarf. This is interesting because the two
// split compile units will have the same offset (0).
// RUN: %clang -g -c -o %t-1.o --target=x86_64-pc-linux -gsplit-dwarf %s
// RUN: %clang -g -c -o %t-2.o --target=x86_64-pc-linux -gsplit-dwarf %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// RUN: %clang -g -c -o %t-1.o --target=x86_64-pc-linux -mllvm -accel-tables=Dwarf %s
// RUN: %clang -g -c -o %t-2.o --target=x86_64-pc-linux -mllvm -accel-tables=Dwarf %S/Inputs/find-variable-file-2.cpp
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --file=find-variable-file.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=ONE %s
// RUN: lldb-test symbols --file=find-variable-file-2.cpp --find=variable %t | \
// RUN:   FileCheck --check-prefix=TWO %s

// ONE: Found 1 variables:
namespace one {
int foo;
// ONE-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-variable-file.cpp:[[@LINE-1]]
} // namespace one

extern "C" void _start() {}

// TWO: Found 1 variables:
// TWO-DAG: name = "foo", {{.*}} decl = find-variable-file-2.cpp:2
