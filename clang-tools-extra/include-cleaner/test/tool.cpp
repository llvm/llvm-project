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

//        RUN: clang-include-cleaner -print=changes %s --ignore-headers="foobar\.h,foo\.h" -- -I%S/Inputs/ | FileCheck --match-full-lines --allow-empty --check-prefix=IGNORE %s
// IGNORE-NOT: - "foobar.h"
// IGNORE-NOT: + "foo.h"

//        RUN: clang-include-cleaner -print=changes %s --ignore-headers="foobar.*\.h" -- -I%S/Inputs/ | FileCheck --match-full-lines --allow-empty --check-prefix=IGNORE2 %s
// IGNORE2-NOT: - "foobar.h"
//     IGNORE2: + "foo.h"

//        RUN: clang-include-cleaner -print=changes %s --ignore-headers= -- -I%S/Inputs/ | FileCheck --allow-empty --check-prefix=IGNORE3 %s
//    IGNORE3: - "foobar.h"
//    IGNORE3: + "foo.h"

//        RUN: clang-include-cleaner -print=changes %s --only-headers="foo\.h" -- -I%S/Inputs/ | FileCheck --match-full-lines --allow-empty --check-prefix=ONLY %s
//   ONLY-NOT: - "foobar.h"
//       ONLY: + "foo.h"

//        RUN: clang-include-cleaner -print=changes %s --only-headers= -- -I%S/Inputs/ | FileCheck --allow-empty --check-prefix=ONLY2 %s
//      ONLY2: - "foobar.h"
//      ONLY2: + "foo.h"

//        RUN: clang-include-cleaner -print %s -- -I%S/Inputs/ | FileCheck --match-full-lines --check-prefix=PRINT %s
//      PRINT: #include "foo.h"
//  PRINT-NOT: {{^}}#include "foobar.h"{{$}}

//        RUN: cp %s %t.cpp
//        RUN: clang-include-cleaner -edit %t.cpp -- -I%S/Inputs/
//        RUN: FileCheck --match-full-lines --check-prefix=EDIT %s < %t.cpp
//       EDIT: #include "foo.h"
//   EDIT-NOT: {{^}}#include "foobar.h"{{$}}

//        RUN: cp %s %t.cpp
//        RUN: clang-include-cleaner -edit --ignore-headers="foobar\.h,foo\.h" %t.cpp -- -I%S/Inputs/ 
//        RUN: FileCheck --match-full-lines --check-prefix=EDIT2 %s < %t.cpp
//  EDIT2-NOT: {{^}}#include "foo.h"{{$}}

// This test has commands that rely on shell capabilities that won't execute
// correctly on Windows e.g. subshell execution
//   REQUIRES: shell
//        RUN: mkdir -p $(dirname %t)/out
//        RUN: cp %s %t.cpp
//        RUN: echo "[{\"directory\":\"$(dirname %t)/out\",\"file\":\"../$(basename %t).cpp\",\"command\":\":clang++ -I%S/Inputs/ ../$(basename %t).cpp\"}]" > $(dirname %t)/out/compile_commands.json
//        RUN: pushd $(dirname %t)
//        RUN: clang-include-cleaner -p out -edit $(basename %t).cpp
//        RUN: popd
//        RUN: FileCheck --match-full-lines --check-prefix=EDIT3 %s < %t.cpp
//      EDIT3: #include "foo.h"
//  EDIT3-NOT: {{^}}#include "foobar.h"{{$}}
