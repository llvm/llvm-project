! Check that flang -fc1 is invoked when in --driver-mode=flang.

! This is a copy of flang_ucase.F90 because the driver has logic in it which
! differentiates between F90 and f90 files. Flang will not treat these files
! differently.

! Test various output types:
! * -E
! * -fsyntax-only
! * -emit-llvm -S
! * -emit-llvm
! * -S
! * (no type specified, resulting in an object file)

! All invocations should begin with flang -fc1, consume up to here.
! ALL-LABEL: "{{[^"]*}}flang-new" "-fc1"

! Check that f90 files are not treated as "previously preprocessed"
! ... in --driver-mode=flang.
! RUN: %clang --driver-mode=flang -### -E                  %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-E %s
! CHECK-E-NOT: previously preprocessed input
! CHECK-E-DAG: "-E"
! CHECK-E-DAG: "-o" "-"

! RUN: %clang --driver-mode=flang -### -emit-ast           %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-EMIT-AST %s
! CHECK-EMIT-AST-DAG: "-emit-ast"
! CHECK-EMIT-AST-DAG: "-o" "{{[^"]*}}.ast"

! RUN: %clang --driver-mode=flang -### -fsyntax-only       %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-SYNTAX-ONLY %s
! CHECK-SYNTAX-ONLY-NOT: "-o"
! CHECK-SYNTAX-ONLY-DAG: "-fsyntax-only"

! RUN: %clang --driver-mode=flang -### -emit-llvm -S       %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-EMIT-LLVM-IR %s
! CHECK-EMIT-LLVM-IR-DAG: "-emit-llvm"
! CHECK-EMIT-LLVM-IR-DAG: "-o" "{{[^"]*}}.ll"

! RUN: %clang --driver-mode=flang -### -emit-llvm          %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-EMIT-LLVM-BC %s
! CHECK-EMIT-LLVM-BC-DAG: "-emit-llvm-bc"
! CHECK-EMIT-LLVM-BC-DAG: "-o" "{{[^"]*}}.bc"

! RUN: %clang --driver-mode=flang -### -S                  %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-S %s
! CHECK-S-DAG: "-S"
! CHECK-S-DAG: "-o" "{{[^"]*}}.s"

! RUN: %clang --driver-mode=flang -### -fintegrated-as     %s 2>&1 | FileCheck --check-prefixes=ALL,CHECK-EMIT-OBJ %s
! CHECK-EMIT-OBJ-DAG: "-emit-obj"
! CHECK-EMIT-OBJ-DAG: "-o" "{{[^"]*}}.o"

! Should end in the input file.
! ALL: "{{.*}}flang.f90"{{$}}
