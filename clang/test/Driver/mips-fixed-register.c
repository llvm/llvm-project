// RUN: %clang --target=mipsel-unknown-linux-gnu \
// RUN:   -ffixed-1 -ffixed-2 -ffixed-3 -ffixed-4 -ffixed-5 \
// RUN:   -ffixed-6 -ffixed-7 -ffixed-8 -ffixed-9 -ffixed-10 \
// RUN:   -ffixed-11 -ffixed-12 -ffixed-13 -ffixed-14 -ffixed-15 \
// RUN:   -ffixed-16 -ffixed-17 -ffixed-18 -ffixed-19 -ffixed-20 \
// RUN:   -ffixed-21 -ffixed-22 -ffixed-23 -ffixed-24 -ffixed-25 \
// RUN:   -ffixed-26 -ffixed-27 -ffixed-28 -ffixed-29 -ffixed-30 \
// RUN:   -ffixed-31 -### %s 2>&1 | FileCheck %s

// CHECK-DAG: "-target-feature" "+reserve-gpr1"
// CHECK-DAG: "-target-feature" "+reserve-gpr2"
// CHECK-DAG: "-target-feature" "+reserve-gpr3"
// CHECK-DAG: "-target-feature" "+reserve-gpr4"
// CHECK-DAG: "-target-feature" "+reserve-gpr5"
// CHECK-DAG: "-target-feature" "+reserve-gpr6"
// CHECK-DAG: "-target-feature" "+reserve-gpr7"
// CHECK-DAG: "-target-feature" "+reserve-gpr8"
// CHECK-DAG: "-target-feature" "+reserve-gpr9"
// CHECK-DAG: "-target-feature" "+reserve-gpr10"
// CHECK-DAG: "-target-feature" "+reserve-gpr11"
// CHECK-DAG: "-target-feature" "+reserve-gpr12"
// CHECK-DAG: "-target-feature" "+reserve-gpr13"
// CHECK-DAG: "-target-feature" "+reserve-gpr14"
// CHECK-DAG: "-target-feature" "+reserve-gpr15"
// CHECK-DAG: "-target-feature" "+reserve-gpr16"
// CHECK-DAG: "-target-feature" "+reserve-gpr17"
// CHECK-DAG: "-target-feature" "+reserve-gpr18"
// CHECK-DAG: "-target-feature" "+reserve-gpr19"
// CHECK-DAG: "-target-feature" "+reserve-gpr20"
// CHECK-DAG: "-target-feature" "+reserve-gpr21"
// CHECK-DAG: "-target-feature" "+reserve-gpr22"
// CHECK-DAG: "-target-feature" "+reserve-gpr23"
// CHECK-DAG: "-target-feature" "+reserve-gpr24"
// CHECK-DAG: "-target-feature" "+reserve-gpr25"
// CHECK-DAG: "-target-feature" "+reserve-gpr26"
// CHECK-DAG: "-target-feature" "+reserve-gpr27"
// CHECK-DAG: "-target-feature" "+reserve-gpr28"
// CHECK-DAG: "-target-feature" "+reserve-gpr29"
// CHECK-DAG: "-target-feature" "+reserve-gpr30"
// CHECK-DAG: "-target-feature" "+reserve-gpr31"

// RUN: not %clang --target=x86_64-unknown-linux-gnu -ffixed-24 -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOT-MIPS %s
// NOT-MIPS: error: unsupported option '-ffixed-24' for target 'x86_64-unknown-linux-gnu'
