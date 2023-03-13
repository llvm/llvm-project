! RUN: %flang -### %s 2>&1 | FileCheck %s

! CHECK: "{{.*}}flang-new" "-fc1" {{.*}} "-o" "{{.*}}.o"
program f08
end program f08
