// Check -ftime-report/-ftime-report= output
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report %s -o /dev/null \
// RUN:     -mllvm -info-output-file=%t
// RUN: cat %t | FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report=per-pass %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report=per-pass-run %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=TIME,NPM-PER-INVOKE
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report-json %s -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=JSON
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     -ftime-report-json %s -o /dev/null \
// RUN:     -mllvm -info-output-file=%t
// RUN: cat %t | FileCheck %s --check-prefixes=JSON
// Check that -stats-file-timers only outputs pass time info in the stats file
// and not stderr.
// RUN: %clang_cc1 -emit-obj -O1 \
// RUN:     %s -o /dev/null \
// RUN:     -stats-file=%t -stats-file-timers 2>&1 | count 0
// RUN: FileCheck %s -input-file=%t -check-prefixes=JSON

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
// JSON:{
// JSON: "time.pass.InstCombinePass.wall": {{.*}},
// JSON: "time.pass.InstCombinePass.user": {{.*}},
// JSON: "time.pass.InstCombinePass.sys": {{.*}},
// JSON:}

int foo(int x, int y) { return x + y; }
