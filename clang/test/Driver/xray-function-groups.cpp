// RUN: %clang -### --target=aarch64-unknown-linux-gnu -fxray-instrument -fxray-function-groups=3 %s 2>&1 | FileCheck %s --check-prefix=GROUP0
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fxray-instrument -fxray-function-groups=3 -fxray-selected-function-group=0 %s 2>&1 | FileCheck %s --check-prefix=GROUP0
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fxray-instrument -fxray-function-groups=3 -fxray-selected-function-group=1 %s 2>&1 | FileCheck %s --check-prefix=GROUP1

// GROUP0: "-fxray-function-groups=3"
// GROUP0-NOT: "-fxray-selected-function-group=
// GROUP1: "-fxray-function-groups=3" "-fxray-selected-function-group=1"
