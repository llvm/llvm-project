// RUN: not %clang %s -S -o - 2>&1 | FileCheck %s
// CHECK: In file included from {{.*}}SourceLocationsOverflow.c
// CHECK-NEXT: inc1.h{{.*}}: fatal error: sorry, this include generates a translation unit too large for Clang to process.
// CHECK-NEXT: #include "inc2.h"
// CHECK-NEXT:          ^
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
