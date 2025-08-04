// RUN: not %clang %s -S -o - 2>&1 | FileCheck %s
// CHECK: In file included from {{.*}}SourceLocationsOverflow.c
// CHECK-NEXT: inc1.h{{.*}}: fatal error: translation unit is too large for Clang to process: ran out of source locations
// CHECK-NEXT: #include "inc2.h"
// CHECK-NEXT:          ^
// CHECK-NEXT: note: 214{{.......}}B (2.15GB) in local locations, 0B (0B) in locations loaded from AST files, for a total of 214{{.......}}B (2.15GB) (99% of available space)
// CHECK-NEXT: {{.*}}inc2.h:1:1: note: file entered 214{{..}} times using 214{{.......}}B (2.15GB) of space
// CHECK-NEXT: /*.................................................................................................
// CHECK-NEXT: ^
// CHECK-NEXT: {{.*}}inc1.h:1:1: note: file entered 15 times using 39{{....}}B (396.92kB) of space
// CHECK-NEXT: #include "inc2.h"
// CHECK-NEXT: ^
// CHECK-NEXT: <built-in>:1:1: note: file entered {{.*}} times using {{.*}}B ({{.*}}B) of space
// CHECK-NEXT: # {{.*}}
// CHECK-NEXT: ^
// CHECK-NEXT: {{.*}}SourceLocationsOverflow.c:1:1: note: file entered 1 time using {{.*}}B ({{.*}}B) of space
// CHECK-NEXT: // RUN: not %clang %s -S -o - 2>&1 | FileCheck %s
// CHECK-NEXT: ^
// CHECK-NEXT: 1 error generated.
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
#include "Inputs/inc1.h"
