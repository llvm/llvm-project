// RUN: %clang -### -c --target=x86_64 -mfpmath=sse %s 2>&1 | FileCheck %s
// CHECK: "-mfpmath" "sse"

/// Don't warn for assembler input.
// RUN: %clang -### -Werror -c --target=x86_64 -mfpmath=sse -x assembler %s 2>&1 | FileCheck /dev/null --implicit-check-not='"-mfpmath"'
