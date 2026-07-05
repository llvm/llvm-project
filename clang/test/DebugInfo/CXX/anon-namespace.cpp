// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -dwarf-explicit-import -O0 %s -o - | FileCheck --check-prefix=IMPORT %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -O0 %s -o - | FileCheck --check-prefix=NOIMPORT %s

namespace
{
  int a = 5;
}
int *b = &a;

namespace
{
  namespace {
    int a1 = 5;
  }
  int a2 = 7;
}
int *b1 = &a1;
int *b2 = &a2;

// IMPORT:  [[NS:![0-9]+]] = !DINamespace
// IMPORT:  [[CU:![0-9]+]] = distinct !DICompileUnit
// IMPORT:  [[NS2:![0-9]+]] = !DINamespace
// IMPORT: !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[CU]], entity: [[NS]], file: {{![0-9]+}})
// IMPORT: !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[NS]], entity: [[NS2]], file: {{![0-9]+}}, line: {{[0-9]+}})
// NOIMPORT-NOT: !DIImportedEntity
