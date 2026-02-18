// RUN: %check_clang_tidy %s readability-duplicate-include %t -- \
// RUN:   -config="{CheckOptions: {readability-duplicate-include.IgnoredFilesList: 'pack_.*\\.h'}}" \
// RUN:   -header-filter='' -- -I %S/Inputs/duplicate-include

int g;
#include "duplicate-include.h"
int h;
#include "duplicate-include.h"
int i;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include [readability-duplicate-include]
// CHECK-FIXES:      int g;
// CHECK-FIXES-NEXT: #include "duplicate-include.h"
// CHECK-FIXES-NEXT: int h;
// CHECK-FIXES-NEXT: int i;

#include "pack_begin.h"
struct A { int x; };
#include "pack_end.h"

#include "pack_begin.h"
struct B { int x; };
#include "pack_end.h"

// No warning here.
