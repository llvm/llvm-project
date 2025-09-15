// RUN: %flang -### -S %s -g -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF5 %s
// RUN: %flang -### -S %s -g1 -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G1-DWARF5 %s
// RUN: %flang -### -S %s -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITHOUT-G-DWARF5 %s
// RUN: %flang -### -S %s -gdwarf-4  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF4 %s
// RUN: %flang -### -S %s -gdwarf-3  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF3 %s
// RUN: %flang -### -S %s -gdwarf-2  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF2 %s

// CHECK-WITH-G-DWARF5: -debug-info-kind=standalone
// CHECK-WITH-G-DWARF5-SAME: -dwarf-version=5

// CHECK-WITH-G1-DWARF5: -debug-info-kind=line-tables-only
// CHECK-WITH-G1-DWARF5-SAME: -dwarf-version=5

// CHECK-WITHOUT-G-DWARF5: -debug-info-kind=standalone
// CHECK-WITHOUT-G-DWARF5-SAME: -dwarf-version=5

// CHECK-DWARF4: -dwarf-version=4

// CHECK-DWARF3: -dwarf-version=3

// CHECK-DWARF2: -dwarf-version=2
