// RUN: %clang -### -target powerpc-ibm-aix -fms-anonymous-structs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MS-STRUCT-ENABLE
// MS-STRUCT-ENABLE: "-fms-anonymous-structs"

// RUN: %clang -### -target powerpc-ibm-aix -fno-ms-anonymous-structs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MS-STRUCT-DISABLE
// MS-STRUCT-DISABLE: "-fno-ms-anonymous-structs"

// RUN: %clang -### -target powerpc-ibm-aix -fms-extensions %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=IMPLICIT-ENABLE
// IMPLICIT-ENABLE: "-fms-anonymous-structs"

// RUN: %clang -### -target powerpc-ibm-aix -fms-compatibility %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=IMPLICIT-ENABLE

// RUN: %clang -### -target powerpc-ibm-aix -fno-ms-anonymous-structs -fms-anonymous-structs %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=LAST-ENABLE
// LAST-ENABLE: "-fms-anonymous-structs"
// LAST-ENABLE-NOT: "-fno-ms-anonymous-structs"

// RUN: %clang -### -target powerpc-ibm-aix -fno-ms-anonymous-structs -fms-extensions %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=LAST-ENABLE

// RUN: %clang -### -target powerpc-ibm-aix -fms-anonymous-structs -fno-ms-anonymous-structs %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=LAST-DISABLE
// LAST-DISABLE: "-fno-ms-anonymous-structs"
// LAST-DISABLE-NOT: "-fms-anonymous-structs"

// RUN: %clang -### -target powerpc-ibm-aix -fms-extensions -fno-ms-anonymous-structs %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=LAST-DISABLE

// RUN: %clang -### -target powerpc-ibm-aix -fms-compatibility -fno-ms-anonymous-structs %s 2>&1 |\
// RUN:   FileCheck %s --check-prefix=LAST-DISABLE

// RUN: %clang -### -target powerpc-ibm-aix %s 2>&1 | FileCheck %s --check-prefix=NO-MS-STRUCT
// NO-MS-STRUCT-NOT: "-fms-anonymous-structs"
// NO-MS-STRUCT-NOT: "-fno-ms-anonymous-structs"

