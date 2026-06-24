// RUN: not %clang --target=armv7a-none-eabi -mlong-calls -mexecute-only -fPIC -### %s 2>&1 | FileCheck --check-prefix=ERR-XO-PIC %s
// ERR-XO-PIC: error: '-mlong-calls' with '-mexecute-only' is not supported for position-independent code

// RUN: not %clang --target=armv7a-none-eabi -mlong-calls -mexecute-only -fPIE -### %s 2>&1 | FileCheck --check-prefix=ERR-XO-PIE %s
// ERR-XO-PIE: error: '-mlong-calls' with '-mexecute-only' is not supported for position-independent code

// RUN: %clang --target=armv7a-none-eabi -mlong-calls -mexecute-only -### %s 2>&1 | FileCheck --check-prefix=OK-XO %s
// OK-XO-NOT: error:

// RUN: not %clang --target=armv7a-none-eabi -mlong-calls -fropi -### %s 2>&1 | FileCheck --check-prefix=ERR-ROPI %s
// ERR-ROPI: error: '-mlong-calls' is not supported with ROPI

// RUN: not %clang --target=armv7a-none-eabi -mlong-calls -frwpi -### %s 2>&1 | FileCheck --check-prefix=ERR-RWPI %s
// ERR-RWPI: error: '-mlong-calls' is not supported with RWPI

// RUN: %clang --target=armv7a-none-eabi -mlong-calls -fropi -mno-long-calls -### %s 2>&1 | FileCheck --check-prefix=OK %s
// OK-NOT: error:
