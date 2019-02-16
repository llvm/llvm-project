// REQUIRES: clang-driver
// XFAIL: windows-msvc

// RUN: %clang -### -S -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s

// RUN: %clang -### -S -fprofile-arcs %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s

// RUN: %clang -### -S -fprofile-arcs %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -c -fprofile-arcs %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s


// CHECK-GCNO-DEFAULT-LOCATION: "-coverage-notes-file" "{{.*}}{{/|\\\\}}coverage_no_integrated_as.c"
// CHECK-GCNO-DEFAULT-LOCATION-NOT: "-coverage-notes-file" "/tmp/{{.*}}/coverage_no_integrated_as.c"
// CHECK-GCNO-LOCATION: "-coverage-notes-file" "{{.*}}/foo/bar.gcno"
// CHECK-GCNO-LOCATION-REL-PATH: "-coverage-notes-file" "{{.*}}{{/|\\\\}}foo/bar.gcno"
