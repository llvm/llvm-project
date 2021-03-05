// This test checks if C functions with internal linkage names are mangled
// and the module hash suffixes attached including emitting DW_AT_linkage_name.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -emit-llvm -o -  %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -funique-internal-linkage-names -emit-llvm -o -  %s | FileCheck %s --check-prefix=UNIQUE
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -emit-llvm -o -  %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -funique-internal-linkage-names -emit-llvm -o -  %s | FileCheck %s --check-prefix=UNIQUE

static int glob;
static int foo(void) {
  return glob;
}

void baz() {
  foo();
}

// PLAIN: @glob = internal global i32
// PLAIN: define internal i32 @foo()
// PLAIN: distinct !DIGlobalVariable(name: "glob"{{.*}})
// PLAIN: distinct !DISubprogram(name: "foo"{{.*}})
// PLAIN-NOT: linkageName:
//
// UNIQUE: @_ZL4glob.[[MODHASH:__uniq.[0-9]+]] = internal global i32
// UNIQUE: define internal i32 @_ZL3foov.[[MODHASH]]()
// UNIQUE: distinct !DIGlobalVariable(name: "glob", linkageName: "_ZL4glob.[[MODHASH]]"{{.*}})
// UNIQUE: distinct !DISubprogram(name: "foo", linkageName: "_ZL3foov.[[MODHASH]]"{{.*}})
