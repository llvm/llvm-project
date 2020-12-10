// Check that legacy pass manager could only use -ftime-report
// RUN: %clang_cc1 -flegacy-pass-manager -emit-obj -O1 \
// RUN:     -ftime-report %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,LPM
// RUN: not %clang_cc1 -flegacy-pass-manager -emit-obj -O1 \
// RUN:     -ftime-report=per-pass %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=ERROR
// RUN: not %clang_cc1 -flegacy-pass-manager -emit-obj -O1 \
// RUN:     -ftime-report=per-pass-run %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=ERROR

// Check -ftime-report/-ftime-report= output for the new pass manager
// RUN: %clang_cc1 -emit-obj -O1 -fno-legacy-pass-manager \
// RUN:     -ftime-report %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 -fno-legacy-pass-manager \
// RUN:     -ftime-report=per-pass %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 -fno-legacy-pass-manager \
// RUN:     -ftime-report=per-pass-run %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM-PER-INVOKE

// TIME: Pass execution timing report
// TIME: Total Execution Time:
// TIME: Name
// LPM-DAG:   Dominator Tree Construction #
// LPM-DAG:   Dominator Tree Construction #
// LPM-DAG:   Dominator Tree Construction #
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-NOT:   InstCombinePass #
// NPM:       InstCombinePass{{$}}
// NPM-NOT:   InstCombinePass #
// TIME: Total{{$}}
// LPM-NOT: Pass execution timing report
// NPM: Pass execution timing report

// ERROR: error: invalid argument '-ftime-report={{.*}}' only allowed with '-fno-legacy-pass-manager'

int foo(int x, int y) { return x + y; }
