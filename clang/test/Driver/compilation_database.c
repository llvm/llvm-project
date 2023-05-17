// RUN: mkdir -p %t.workdir && cd %t.workdir
// RUN: %clang -no-canonical-prefixes -fintegrated-as -MD -MP --sysroot=somewhere -c -x c %s -xc++ %s -Wall -MJ - 2>&1 | FileCheck %s
// RUN: not %clang -no-canonical-prefixes -c -x c %s -MJ %s/non-existant 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: { "directory": "{{[^"]*}}workdir",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc", "[[SRC]]", "-o", "compilation_database.o", "-no-canonical-prefixes", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// CHECK: { "directory": "{{.*}}",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc++", "[[SRC]]", "-o", "compilation_database.o", "-no-canonical-prefixes", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// ERROR: error: compilation database '{{.*}}/non-existant' could not be opened:

int main(void) {
  return 0;
}
