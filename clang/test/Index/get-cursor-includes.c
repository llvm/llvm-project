#include "get-cursor-includes-2.h"
#include "get-cursor-includes-2.h"

// RUN: c-index-test -write-pch %t.h.pch -I%S/Inputs -Xclang -detailed-preprocessing-record %S/Inputs/get-cursor-includes-2.h
// RUN: c-index-test -cursor-at=%S/Inputs/get-cursor-includes-2.h:1:5 -I%S/Inputs -include %t.h %s | FileCheck %s

// CHECK: inclusion directive=get-cursor-includes-1.h
