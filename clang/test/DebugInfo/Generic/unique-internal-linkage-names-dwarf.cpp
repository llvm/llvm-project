// This test checks if C++ functions with internal linkage names are mangled
// and the module hash suffixes attached including emitting DW_AT_linkage_name.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -emit-llvm -o - %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -funique-internal-linkage-names -emit-llvm -o - %s | FileCheck %s --check-prefix=UNIQUE
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -emit-llvm -o - %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -funique-internal-linkage-names -emit-llvm -o - %s | FileCheck %s --check-prefix=UNIQUE

static int glob_foo = 5;
static int foo(void) {
  __builtin_printf("%p", &glob_foo);
  return glob_foo;
}

// Anonymous namespaces generate internal linkage symbols.
namespace {
  int glob_bar;
  int bar() {
    return glob_bar;
  }
}

extern "C" {
  static int glob_zip;
  static int zip(void) {
    return glob_zip;
  }
}

void baz() {
  foo();
  bar();
  zip();
}

// PLAIN-DAG: @_ZL8glob_foo = internal global i32
// PLAIN-DAG: define internal noundef i32 @_ZL3foov()
// PLAIN-DAG: distinct !DIGlobalVariable(name: "glob_foo", linkageName: "_ZL8glob_foo"{{.*}})
// PLAIN-DAG: distinct !DISubprogram(name: "foo", linkageName: "_ZL3foov"{{.*}})
// PLAIN-DAG: @_ZN12_GLOBAL__N_18glob_barE = internal global i32
// PLAIN-DAG: define internal noundef i32 @_ZN12_GLOBAL__N_13barEv()
// PLAIN-DAG: distinct !DIGlobalVariable(name: "glob_bar", linkageName: "_ZN12_GLOBAL__N_18glob_barE"{{.*}})
// PLAIN-DAG: distinct !DISubprogram(name: "bar", linkageName: "_ZN12_GLOBAL__N_13barEv"{{.*}})
// PLAIN-DAG: @_ZL8glob_zip = internal global i32
// PLAIN-DAG: define internal noundef i32 @_ZL3zipv()
// PLAIN-DAG: distinct !DIGlobalVariable(name: "glob_zip", linkageName: "_ZL8glob_zip"{{.*}})
// PLAIN-DAG: distinct !DISubprogram(name: "zip", linkageName: "_ZL3zipv"{{.*}})

// UNIQUE-DAG: @_ZL8glob_foo = internal global i32
// UNIQUE-DAG: define internal noundef i32 @_ZL3foov.[[MODHASH:__uniq\.[0-9]+]]()
// UNIQUE-DAG: distinct !DIGlobalVariable(name: "glob_foo", linkageName: "_ZL8glob_foo"{{.*}})
// UNIQUE-DAG: distinct !DISubprogram(name: "foo", linkageName: "_ZL3foov.[[MODHASH]]"{{.*}})
// UNIQUE-DAG: @_ZN12_GLOBAL__N_18glob_barE = internal global i32
// UNIQUE-DAG: define internal noundef i32 @_ZN12_GLOBAL__N_13barEv.[[MODHASH]]()
// UNIQUE-DAG: distinct !DIGlobalVariable(name: "glob_bar", linkageName: "_ZN12_GLOBAL__N_18glob_barE"{{.*}})
// UNIQUE-DAG: distinct !DISubprogram(name: "bar", linkageName: "_ZN12_GLOBAL__N_13barEv.[[MODHASH]]"{{.*}})
// UNIQUE-DAG: @_ZL8glob_zip = internal global i32
// UNIQUE-DAG: define internal noundef i32 @_ZL3zipv.[[MODHASH]]()
// UNIQUE-DAG: distinct !DIGlobalVariable(name: "glob_zip", linkageName: "_ZL8glob_zip"{{.*}})
// UNIQUE-DAG: distinct !DISubprogram(name: "zip", linkageName: "_ZL3zipv.[[MODHASH]]"{{.*}})
