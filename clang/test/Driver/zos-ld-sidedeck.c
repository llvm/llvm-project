// Try using various forms of output file name to see what side deck file name looks like
// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o foo.out 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-BASE %s
// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o foo 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-BASE %s
// SD-BASE: "-x" "foo.x"

// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o lib/foo.out 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-SUBDIR %s
// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o lib/foo 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-SUBDIR %s
// SD-SUBDIR: "-x" "lib/foo.x"


// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o ../lib/foo.out 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-REL %s
// RUN: %clang -### --shared --target=s390x-ibm-zos %s -o ../lib/foo 2>&1 \
// RUN:   | FileCheck --check-prefix=SD-REL %s
// SD-REL: "-x" "../lib/foo.x"
