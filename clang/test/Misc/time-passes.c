// Check -ftime-report/-ftime-report= output
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report=per-pass %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report=per-pass-run %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM-PER-INVOKE

// TIME: Pass execution timing report
// TIME: Total Execution Time:
// TIME: Name
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-PER-INVOKE-DAG:   InstCombinePass #
// NPM-NOT:   InstCombinePass #
// NPM:       InstCombinePass{{$}}
// NPM-NOT:   InstCombinePass #
// TIME: Total{{$}}
// NPM: Pass execution timing report

int foo(int x, int y) { return x + y; }
