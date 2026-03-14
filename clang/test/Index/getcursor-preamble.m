#include "getcursor-preamble.h"

// RUN: c-index-test \
// RUN:    -cursor-at=%S/getcursor-preamble.h:2:10 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:3:9 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:4:6 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:5:8 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:7:7 \
// RUN:             %s | FileCheck %s

// RUN: env CINDEXTEST_EDITING=1 c-index-test \
// RUN:    -cursor-at=%S/getcursor-preamble.h:2:10 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:3:9 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:4:6 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:5:8 \
// RUN:    -cursor-at=%S/getcursor-preamble.h:7:7 \
// RUN:             %s | FileCheck %s

// CHECK: StructDecl=AA:2:10
// CHECK: FieldDecl=x:3:9
// CHECK: ObjCIvarDecl=aa:4:5
// CHECK: ObjCIvarDecl=var:5:7
// CHECK: ObjCInstanceMethodDecl=foo:7:6
