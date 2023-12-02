! RUN: %flang -### %s 2>&1 | FileCheck %s

! CHECK: "{{.*}}flang-{{[0-9]+}}" "-fc1" {{.*}} "-o" "{{.*}}.o"
program f03
end program f03
