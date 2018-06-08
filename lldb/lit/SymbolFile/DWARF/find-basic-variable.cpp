// REQUIRES: lld

// RUN: clang %s -g -c -o %t.o --target=x86_64-pc-linux
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=variable --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --regex --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=REGEX %s
// RUN: lldb-test symbols --name=not_there --find=variable %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s
//
// RUN: clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=variable --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --regex --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=REGEX %s
// RUN: lldb-test symbols --name=not_there --find=variable %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s
//
// RUN: clang %s -g -c -emit-llvm -o - --target=x86_64-pc-linux | \
// RUN:   llc -accel-tables=Dwarf -filetype=obj -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=variable --context=context %t | \
// RUN:   FileCheck --check-prefix=CONTEXT %s
// RUN: lldb-test symbols --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=NAME %s
// RUN: lldb-test symbols --regex --name=foo --find=variable %t | \
// RUN:   FileCheck --check-prefix=REGEX %s
// RUN: lldb-test symbols --name=not_there --find=variable %t | \
// RUN:   FileCheck --check-prefix=EMPTY %s

// EMPTY: Found 0 variables:
// NAME: Found 4 variables:
// CONTEXT: Found 1 variables:
// REGEX: Found 5 variables:
int foo;
// NAME-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-1]]
// REGEX-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-2]]
namespace bar {
int context;
long foo;
// NAME-DAG: name = "foo", type = {{.*}} (long int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-1]]
// CONTEXT-DAG: name = "foo", type = {{.*}} (long int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-2]]
// REGEX-DAG: name = "foo", type = {{.*}} (long int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-3]]
namespace baz {
static short foo;
// NAME-DAG: name = "foo", type = {{.*}} (short), {{.*}} decl = find-basic-variable.cpp:[[@LINE-1]]
// REGEX-DAG: name = "foo", type = {{.*}} (short), {{.*}} decl = find-basic-variable.cpp:[[@LINE-2]]
}
}

struct sbar {
  static int foo;
// NAME-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-1]]
// REGEX-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-2]]
};
int sbar::foo;

int foobar;
// REGEX-DAG: name = "foobar", type = {{.*}} (int), {{.*}} decl = find-basic-variable.cpp:[[@LINE-1]]

int fbar() {
  static int foo;
  return foo + bar::baz::foo;
}

int Foo;

struct ssbar {
  int foo;
};

extern "C" void _start(sbar, ssbar) {}
