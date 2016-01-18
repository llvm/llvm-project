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
