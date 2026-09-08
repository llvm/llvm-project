// RUN: %clang_cc1 -O1 -disable-llvm-passes -gcall-site-info -dwarf-version=5 -emit-llvm \
// RUN:    -debug-info-kind=constructor -triple x86_64-unknown-unknown %s -o - \
// RUN: | FileCheck %s

// Check that DISubprogram metadata that is attached to methods out of
// necessity for call-site-info inclues all the fields/info that would be
// otherwise used for method fwd decls. Tested in this case by checking for the
// presence of `scopeLine`, which would be omitted if the method were treated
// as free function fwd decl.

struct a {
  a();
} b;

// CHECK: !DISubprogram(name: "a", linkageName: "_ZN1aC4Ev", scope: ![[#]], file: ![[#]], line: [[# @LINE - 3]], type: ![[#]], scopeLine: [[# @LINE - 3]], flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
