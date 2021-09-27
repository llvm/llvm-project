// RUN: mkdir -p %t.workdir && cd %t.workdir
// RUN: %clang -fintegrated-as -MD -MP --sysroot=somewhere -c -x c %s -xc++ %s -Wall -MJ - -no-canonical-prefixes 2>&1 | FileCheck %s
// RUN: not %clang -c -x c %s -MJ %s/non-existant -no-canonical-prefixes 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: { "directory": "{{[^"]*}}workdir",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc", "[[SRC]]", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// CHECK: { "directory": "{{.*}}",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc++", "[[SRC]]", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// ERROR: error: compilation database '{{.*}}/non-existant' could not be opened:

int main(void) {
  return 0;
}
