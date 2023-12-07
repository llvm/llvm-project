! RUN: %flang -### %s 2>&1 | FileCheck %s

! CHECK: "{{.*}}flang-new" "-fc1" {{.*}} "-o" "{{.*}}.o"
program f03
end program f03
