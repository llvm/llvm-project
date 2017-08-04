// RUN: sed -e "s:INPUT_DIR:%S/Inputs:g" -e "s:OUT_DIR:%t:g" %S/Inputs/overlay.yaml > %t.yaml
// REQUIRES: shell

#include "using-overlay.h"

// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx -I %t -ivfsoverlay %t.yaml
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s

// XFAIL: linux

// CHECK: Record | user | {{.*}}test/Index/Store/Inputs/using-overlay.h
