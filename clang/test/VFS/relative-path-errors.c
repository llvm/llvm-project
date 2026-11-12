// RUN: mkdir -p %t
// RUN: cd %t
// RUN: cp %s main.c
// RUN: not %clang_cc1 main.c 2>&1 | FileCheck %s
// RUN: echo '{"roots": [],"version": 0}' > %t.yaml
// RUN: not %clang_cc1 -ivfsoverlay %t.yaml main.c 2>&1 | FileCheck %s
// RUN: echo '{"version": 0,"roots":[{"type":"directory","name":"%/t","contents":[{"type":"file","name":"vfsname", "external-contents":"main.c"}]}]}' > %t.yaml
// RUN: not %clang_cc1 -ivfsoverlay %t.yaml vfsname 2>&1 | FileCheck %s

// CHECK: {{^}}main.c:[[# @LINE + 1]]:1: error:
foobarbaz
