// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp "%s" "%t/ctu-on-demand-parsing-ambiguous-compilation-database.c"
// RUN: cp "%S/Inputs/ctu-other.c" "%t/ctu-other.c"
// Path substitutions on Windows platform could contain backslashes. These are escaped in the json file.
// Note there is a duplicate entry for 'ctu-other.c'.
// RUN: echo '[{"directory":"%t","command":"gcc -c -std=c89 -Wno-visibility ctu-other.c","file":"ctu-other.c"},{"directory":"%t","command":"gcc -c -std=c89 -Wno-visibility ctu-other.c","file":"ctu-other.c"}]' | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cd "%t" && %clang_extdef_map ctu-other.c > externalDefMap.txt
// The exit code of the analysis is 1 if the import error occurs
// RUN: cd "%t" && not %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -std=c89 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config ctu-on-demand-parsing=true \
// RUN:   ctu-on-demand-parsing-ambiguous-compilation-database.c 2>&1 | FileCheck %t/ctu-on-demand-parsing-ambiguous-compilation-database.c

// CHECK: {{.*}}multiple definitions are found for the same key in index

// 'int f(int)' is defined in ctu-other.c
int f(int);
void testAmbiguousImport() {
  f(0);
}
