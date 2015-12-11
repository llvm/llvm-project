// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-scei-ps4 -O0 %s -o - | FileCheck --check-prefix=PS4 %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-unknown-linux-gnu -O0 %s -o - | FileCheck --check-prefix=NON-PS4 %s

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


// PS4:  [[NS:![0-9]+]] = !DINamespace
// PS4:  [[NS2:![0-9]+]] = !DINamespace
// PS4: !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0, entity: [[NS]])
// PS4: !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[NS]], entity: [[NS2]], line: {{[0-9]+}})
// NON-PS4-NOT: !DIImportedEntity

