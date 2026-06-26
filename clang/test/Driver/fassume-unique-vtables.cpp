// Both -f[no-]assume-unique-vtables are forwarded to -cc1 (last one wins).

// RUN: %clang -### %s -S 2>&1 | FileCheck %s -check-prefix=DEFAULT
// DEFAULT-NOT: "-fassume-unique-vtables"
// DEFAULT-NOT: "-fno-assume-unique-vtables"

// RUN: %clang -### -fno-assume-unique-vtables %s -S 2>&1 | FileCheck %s -check-prefix=NO --implicit-check-not="-fassume-unique-vtables"
// RUN: %clang -### -fassume-unique-vtables -fno-assume-unique-vtables %s -S 2>&1 | FileCheck %s -check-prefix=NO --implicit-check-not="-fassume-unique-vtables"
// NO: "-fno-assume-unique-vtables"

// RUN: %clang -### -fassume-unique-vtables %s -S 2>&1 | FileCheck %s -check-prefix=YES --implicit-check-not="-fno-assume-unique-vtables"
// RUN: %clang -### -fno-assume-unique-vtables -fassume-unique-vtables %s -S 2>&1 | FileCheck %s -check-prefix=YES --implicit-check-not="-fno-assume-unique-vtables"
// YES: "-fassume-unique-vtables"
