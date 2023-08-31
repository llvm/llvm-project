// RUN: mkdir -p %t.workdir && cd %t.workdir
// RUN: %clang -no-canonical-prefixes -fintegrated-as -MD -MP --sysroot=somewhere -c -x c %s -xc++ %s -Wall -MJ - 2>&1 | FileCheck %s
// RUN: not %clang -no-canonical-prefixes -c -x c %s -MJ %s/non-existant 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: %clang -fsyntax-only %s -MJ - 2>&1 | FileCheck %s -check-prefix=SYNTAX_ONLY

// CHECK: { "directory": "{{[^"]*}}workdir",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc", "[[SRC]]", "-o", "compilation_database.o", "-no-canonical-prefixes", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// CHECK: { "directory": "{{.*}}",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc++", "[[SRC]]", "-o", "compilation_database.o", "-no-canonical-prefixes", "-fintegrated-as", "--sysroot=somewhere", "-c", "-Wall",{{.*}} "--target={{[^"]+}}"]},
// ERROR: error: compilation database '{{.*}}/non-existant' could not be opened:

// SYNTAX_ONLY:      {
// SYNTAX_ONLY-SAME: "directory": "{{[^"]*}}workdir",
// SYNTAX_ONLY-SAME: "file": "{{[^"]*}}compilation_database.c"
// SYNTAX_ONLY-NOT:  "output"
// SYNTAX_ONLY-SAME: "arguments": [
// SYNTAX_ONLY-NOT:    "-o"
// SYNTAX_ONLY-SAME: ]
// SYNTAX_ONLY-SAME: }

int main(void) {
  return 0;
}
