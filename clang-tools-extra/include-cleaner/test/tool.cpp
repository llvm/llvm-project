#include "foobar.h"

int x = foo();

//         RUN: clang-include-cleaner -print=changes %s -- -I%S/Inputs/ | FileCheck --check-prefix=CHANGE %s
//      CHANGE: - "foobar.h"
// CHANGE-NEXT: + "foo.h"

//         RUN: clang-include-cleaner -remove=0 -print=changes %s -- -I%S/Inputs/ | FileCheck --check-prefix=INSERT %s
//  INSERT-NOT: - "foobar.h"
//      INSERT: + "foo.h"

//         RUN: clang-include-cleaner -insert=0 -print=changes %s -- -I%S/Inputs/ | FileCheck --check-prefix=REMOVE %s
//      REMOVE: - "foobar.h"
//  REMOVE-NOT: + "foo.h"

//        RUN: clang-include-cleaner -print %s -- -I%S/Inputs/ | FileCheck --match-full-lines --check-prefix=PRINT %s
//      PRINT: #include "foo.h"
//  PRINT-NOT: {{^}}#include "foobar.h"{{$}}

//        RUN: cp %s %t.cpp
//        RUN: clang-include-cleaner -edit %t.cpp -- -I%S/Inputs/
//        RUN: FileCheck --match-full-lines --check-prefix=EDIT %s < %t.cpp
//       EDIT: #include "foo.h"
//   EDIT-NOT: {{^}}#include "foobar.h"{{$}}
