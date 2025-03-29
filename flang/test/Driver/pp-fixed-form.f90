!RUN: %flang -save-temps -### %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREE
FREE:       "-fc1" {{.*}} "-o" "free-form-test.i" {{.*}} "-x" "f95-cpp-input" "{{.*}}/free-form-test.f90"
FREE-NEXT:  "-fc1" {{.*}} "-ffixed-form" {{.*}} "-x" "f95" "free-form-test.i"

!RUN: %flang -save-temps -### %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXED
FIXED:      "-fc1" {{.*}} "-o" "fixed-form-test.i" {{.*}} "-x" "f95-cpp-input" "{{.*}}/fixed-form-test.f"
FIXED-NEXT: "-fc1" {{.*}} "-ffixed-form" {{.*}} "-x" "f95" "fixed-form-test.i"

!RUN: %flang -save-temps -### -ffree-form %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREE-FLAG
FREE-FLAG:           "-fc1" {{.*}} "-o" "free-form-test.i" {{.*}} "-x" "f95-cpp-input" "{{.*}}/free-form-test.f90"
FREE-FLAG-NEXT:      "-fc1" {{.*}} "-emit-llvm-bc" "-ffree-form"
FREE-FLAG-NOT:       "-ffixed-form"
FREE-FLAG-SAME:      "-x" "f95" "free-form-test.i"

!RUN: %flang -save-temps -### -ffixed-form %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXED-FLAG
FIXED-FLAG:          "-fc1" {{.*}} "-o" "fixed-form-test.i" {{.*}} "-x" "f95-cpp-input" "{{.*}}/fixed-form-test.f"
FIXED-FLAG-NEXT:     "-fc1" {{.*}} "-emit-llvm-bc" "-ffixed-form"
FIXED-FLAG-NOT:      "-ffixed-form"
FIXED-FLAG-SAME:     "-x" "f95" "fixed-form-test.i"
