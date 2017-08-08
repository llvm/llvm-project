// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-DEFAULT %s
// TRAP-DEFAULT: -cc1as
// TRAP-DEFAULT-NOT: "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--trap 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-ON %s
// TRAP-ON: -cc1as
// TRAP-ON: "-target-feature" "+use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--break 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-OFF %s
// TRAP-OFF: -cc1as
// TRAP-OFF: "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--trap,--break 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-BOTH-TRAP-FIRST %s
// TRAP-BOTH-TRAP-FIRST: -cc1as
// TRAP-BOTH-TRAP-FIRST: "-target-feature" "+use-tcc-in-div" "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--break,--trap 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-BOTH-BREAK-FIRST %s
// TRAP-BOTH-BREAK-FIRST: -cc1as
// TRAP-BOTH-BREAK-FIRST: "-target-feature" "-use-tcc-in-div" "-target-feature" "+use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MSOFT-FLOAT-DEFAULT %s
// MSOFT-FLOAT-DEFAULT: -cc1as
// MSOFT-FLOAT-DEFAULT-NOT: "-target-feature" "-soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-msoft-float 2>&1 | \
// RUN:   FileCheck -check-prefix=MSOFT-FLOAT-ON %s
// MSOFT-FLOAT-ON: -cc1as
// MSOFT-FLOAT-ON: "-target-feature" "+soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mhard-float 2>&1 | \
// RUN:   FileCheck -check-prefix=MSOFT-FLOAT-OFF %s
// MSOFT-FLOAT-OFF: -cc1as
// MSOFT-FLOAT-OFF: "-target-feature" "-soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-msoft-float,-mhard-float 2>&1 | \
// RUN:   FileCheck -check-prefix=MSOFT-FLOAT-BOTH-MSOFT-FLOAT-FIRST %s
// MSOFT-FLOAT-BOTH-MSOFT-FLOAT-FIRST: -cc1as
// MSOFT-FLOAT-BOTH-MSOFT-FLOAT-FIRST: "-target-feature" "+soft-float" "-target-feature" "-soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mhard-float,-msoft-float 2>&1 | \
// RUN:   FileCheck -check-prefix=MSOFT-FLOAT-BOTH-MHARD-FLOAT-FIRST %s
// MSOFT-FLOAT-BOTH-MHARD-FLOAT-FIRST: -cc1as
// MSOFT-FLOAT-BOTH-MHARD-FLOAT-FIRST: "-target-feature" "-soft-float" "-target-feature" "+soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips1 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS1 %s
// MIPS1: -cc1as
// MIPS1: "-target-feature" "+mips1"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips2 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS2 %s
// MIPS2: -cc1as
// MIPS2: "-target-feature" "+mips2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips3 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS3 %s
// MIPS3: -cc1as
// MIPS3: "-target-feature" "+mips3"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips4 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS4 %s
// MIPS4: -cc1as
// MIPS4: "-target-feature" "+mips4"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips5 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS5 %s
// MIPS5: -cc1as
// MIPS5: "-target-feature" "+mips5"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips32 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS32 %s
// MIPS32: -cc1as
// MIPS32: "-target-feature" "+mips32"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips32r2 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS32R2 %s
// MIPS32R2: -cc1as
// MIPS32R2: "-target-feature" "+mips32r2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips32r3 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS32R3 %s
// MIPS32R3: -cc1as
// MIPS32R3: "-target-feature" "+mips32r3"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips32r5 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS32R5 %s
// MIPS32R5: -cc1as
// MIPS32R5: "-target-feature" "+mips32r5"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips32r6 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS32R6 %s
// MIPS32R6: -cc1as
// MIPS32R6: "-target-feature" "+mips32r6"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64 %s
// MIPS64: -cc1as
// MIPS64: "-target-feature" "+mips64"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64r2 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64R2 %s
// MIPS64R2: -cc1as
// MIPS64R2: "-target-feature" "+mips64r2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64r3 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64R3 %s
// MIPS64R3: -cc1as
// MIPS64R3: "-target-feature" "+mips64r3"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64r5 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64R5 %s
// MIPS64R5: -cc1as
// MIPS64R5: "-target-feature" "+mips64r5"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64r6 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64R6 %s
// MIPS64R6: -cc1as
// MIPS64R6: "-target-feature" "+mips64r6"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64r2,-mips4 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64R2-MIPS4 %s
// MIPS64R2-MIPS4: -cc1as
// MIPS64R2-MIPS4-NOT: "-target-feature" "+mips64r2"
// MIPS64R2-MIPS4: "-target-feature" "+mips4"
// MIPS64R2-MIPS4-NOT: "-target-feature" "+mips64r2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,-mips64,-mips32,-mips32r2 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS64-MIPS32-MIPS32R2 %s
// MIPS64-MIPS32-MIPS32R2: -cc1as
// MIPS64-MIPS32-MIPS32R2-NOT: "-target-feature" "+mips64"
// MIPS64-MIPS32-MIPS32R2-NOT: "-target-feature" "+mips32"
// MIPS64-MIPS32-MIPS32R2: "-target-feature" "+mips32r2"
// MIPS64-MIPS32-MIPS32R2-NOT: "-target-feature" "+mips64"
// MIPS64-MIPS32-MIPS32R2-NOT: "-target-feature" "+mips32"
