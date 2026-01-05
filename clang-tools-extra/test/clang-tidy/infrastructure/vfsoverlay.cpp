// UNSUPPORTED: system-windows
// RUN: sed -e "s:INPUT_DIR:%S/Inputs/vfsoverlay:g" -e "s:OUT_DIR:%t:g" %S/Inputs/vfsoverlay/vfsoverlay.yaml > %t.yaml
// RUN: clang-tidy %s -checks='-*,modernize-use-nullptr' -vfsoverlay %t.yaml -- -I %t | FileCheck %s

#include "not_real.h"

X *ptr = 0;
// CHECK: warning: use nullptr [modernize-use-nullptr]
