// RUN: %clang -### -S -ftest-coverage %s 2>&1 | FileCheck --check-prefix=TEST-COVERAGE %s
// RUN: %clang -### -S -ftest-coverage -fno-test-coverage %s 2>&1 | FileCheck --check-prefix=NO-TEST-COVERAGE %s

// TEST-COVERAGE: "-coverage-notes-file={{.*}}{{/|\\\\}}coverage.gcno"
// NO-TEST-COVERAGE-NOT: "-coverage-notes-file=

// RUN: %clang -### -S -fprofile-arcs %s 2>&1 | FileCheck --check-prefix=PROFILE-ARCS %s
// RUN: %clang -### -S -fprofile-arcs -fno-profile-arcs %s 2>&1 | FileCheck --check-prefix=NO-PROFILE-ARCS %s

// NO-PROFILE-ARCS-NOT: "-coverage-notes-file=
// PROFILE-ARCS: "-coverage-data-file={{.*}}{{/|\\\\}}coverage.gcda"

// RUN: %clang -### -S -ftest-coverage %s -o /foo/bar.o 2>&1 | FileCheck --check-prefix=GCNO-LOCATION %s
// RUN: %clang_cl -### /c --coverage /Fo/foo/bar.obj -- %s 2>&1 | FileCheck --check-prefix=GCNO-LOCATION %s
// RUN: %clang -### -c -ftest-coverage %s -o foo/bar.o 2>&1 | FileCheck --check-prefix=GCNO-LOCATION-REL %s

// GCNO-LOCATION: "-coverage-notes-file={{.*}}/foo/bar.gcno"
// GCNO-LOCATION-REL: "-coverage-notes-file={{.*}}{{/|\\\\}}foo/bar.gcno"

/// GCC allows PWD to change the paths.
// RUN: %if system-linux %{ env PWD=/proc/self/cwd %clang -### -c --coverage %s -o foo/bar.o 2>&1 | FileCheck --check-prefix=PWD %s %}
// PWD: "-coverage-notes-file=/proc/self/cwd/foo/bar.gcno" "-coverage-data-file=/proc/self/cwd/foo/bar.gcda"

/// Don't warn -Wunused-command-line-argument.
// RUN: %clang -E -Werror --coverage -ftest-coverage -fprofile-arcs %s

/// Test -fprofile-dir=
// RUN: not %clang -S -Werror -fprofile-dir=abc %s
// RUN: not %clang -S -Werror -ftest-coverage -fprofile-dir=abc %s
// RUN: %clang -### -S -fprofile-arcs -fprofile-dir=abc %s 2>&1 | FileCheck --check-prefix=PROFILE-DIR %s
// RUN: %clang -### -S --coverage -fprofile-dir=abc %s 2>&1 | FileCheck --check-prefix=PROFILE-DIR %s

// PROFILE-DIR: "-coverage-data-file=abc

/// These should only get passed if any of --coverage, -ftest-coverage, or
/// -fprofile-arcs is passed.
// RUN: %clang -### -c %s 2>&1 | FileCheck --check-prefix=NO-COV %s
// NO-COV-NOT: "-coverage-notes-file=
// NO-COV-NOT: "-coverage-data-file=

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: mkdir d e f && cp %s d/a.c && touch d/b.c

// RUN: %clang -### --coverage d/a.c d/b.c -o e/x 2>&1 | FileCheck %s --check-prefix=LINK1
// LINK1: -cc1{{.*}} "-coverage-notes-file={{.*}}{{/|\\\\}}e/x-a.gcno" "-coverage-data-file={{.*}}{{/|\\\\}}e/x-a.gcda"
// LINK1: -cc1{{.*}} "-coverage-notes-file={{.*}}{{/|\\\\}}e/x-b.gcno" "-coverage-data-file={{.*}}{{/|\\\\}}e/x-b.gcda"

// RUN: %clang -### --coverage d/a.c d/b.c -o e/x -dumpdir f/g 2>&1 | FileCheck %s --check-prefix=LINK2
// LINK2: -cc1{{.*}} "-coverage-notes-file={{.*}}{{/|\\\\}}f/ga.gcno" "-coverage-data-file={{.*}}{{/|\\\\}}f/ga.gcda"
// LINK2: -cc1{{.*}} "-coverage-notes-file={{.*}}{{/|\\\\}}f/gb.gcno" "-coverage-data-file={{.*}}{{/|\\\\}}f/gb.gcda"

/// GCC allows PWD to change the paths.
// RUN: %if system-linux %{ env PWD=/proc/self/cwd %clang -### --coverage d/a.c d/b.c -o e/x -fprofile-dir=f 2>&1 | FileCheck %s --check-prefix=LINK3 %}
// LINK3: -cc1{{.*}} "-coverage-notes-file=/proc/self/cwd/e/x-a.gcno" "-coverage-data-file=f/proc/self/cwd/e/x-a.gcda"
// LINK3: -cc1{{.*}} "-coverage-notes-file=/proc/self/cwd/e/x-b.gcno" "-coverage-data-file=f/proc/self/cwd/e/x-b.gcda"
