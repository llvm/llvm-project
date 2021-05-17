// This test checks if C functions with internal linkage names are mangled
// and the module hash suffixes attached including emitting DW_AT_linkage_name.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -emit-llvm -o -  %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=4 -funique-internal-linkage-names -emit-llvm -o -  %s | FileCheck %s --check-prefix=UNIQUE
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -emit-llvm -o -  %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -dwarf-version=5 -funique-internal-linkage-names -emit-llvm -o -  %s | FileCheck %s --check-prefix=UNIQUE

static int glob;
// foo should be given a uniquefied name under -funique-internal-linkage-names.
static int foo(void) {
  return glob;
}

// bar should not be given a uniquefied name under -funique-internal-linkage-names, 
// since it doesn't come with valid prototype.
static int bar(a) int a;
{
  return glob + a;
}

// go should be given a uniquefied name under -funique-internal-linkage-names, even 
// if its definition doesn't come with a valid prototype, but the declaration here
// has a prototype.
static int go(int);

void baz() {
  foo();
  bar(1);
  go(2);
}

static int go(a) int a;
{
  return glob + a;
}


// PLAIN: @glob = internal global i32
// PLAIN: define internal i32 @foo()
// PLAIN: define internal i32 @bar(i32 %a)
// PLAIN: distinct !DIGlobalVariable(name: "glob"{{.*}})
// PLAIN: distinct !DISubprogram(name: "foo"{{.*}})
// PLAIN: distinct !DISubprogram(name: "bar"{{.*}})
// PLAIN: distinct !DISubprogram(name: "go"{{.*}})
// PLAIN-NOT: linkageName:
//
// UNIQUE: @glob = internal global i32
// UNIQUE: define internal i32 @_ZL3foov.[[MODHASH:__uniq.[0-9]+]]()
// UNIQUE: define internal i32 @bar(i32 %a)
// UNIQUE: define internal i32 @_ZL2goi.[[MODHASH]](i32 %a)
// UNIQUE: distinct !DIGlobalVariable(name: "glob"{{.*}})
// UNIQUE: distinct !DISubprogram(name: "foo", linkageName: "_ZL3foov.[[MODHASH]]"{{.*}})
// UNIQUE: distinct !DISubprogram(name: "go", linkageName: "_ZL2goi.[[MODHASH]]"{{.*}})
