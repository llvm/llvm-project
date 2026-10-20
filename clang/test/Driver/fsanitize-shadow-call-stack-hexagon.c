// Test that -fsanitize=shadow-call-stack on Hexagon requires the register
// holding the shadow call stack pointer to be reserved, and that -mscs-reg=
// selects which register that is.

// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-FIXED

// RUN: %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -ffixed-r18 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT

/// Reserving some other register does not satisfy the requirement.
// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -ffixed-r19 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-FIXED

/// -mscs-reg= moves the requirement to the selected register.
// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -mscs-reg=r16 -ffixed-r18 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-FIXED-R16

// RUN: %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -mscs-reg=r16 -ffixed-r16 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCS-R16

/// -mscs-reg= is accepted without the sanitizer, and still sets the feature.
// RUN: %clang --target=hexagon-unknown-linux-musl -mscs-reg=r27 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCS-R27

/// Only the callee-saved registers r16-r27 can hold the pointer across calls.
// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -mscs-reg=r15 -ffixed-r15 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-REG

// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -mscs-reg=r28 -ffixed-r28 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-REG-28

// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -mscs-reg=sp -ffixed-r18 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-REG-SP

// NO-FIXED: error: invalid argument '-fsanitize=shadow-call-stack' only allowed with '-ffixed-r18'
// DEFAULT-DAG: "-target-feature" "+reserved-r18"
// DEFAULT-DAG: "-fsanitize=shadow-call-stack"

// NO-FIXED-R16: error: invalid argument '-fsanitize=shadow-call-stack' only allowed with '-ffixed-r16'
// SCS-R16-DAG: "-target-feature" "+reserved-r16"
// SCS-R16-DAG: "-target-feature" "+scs-reg-r16"
// SCS-R16-DAG: "-fsanitize=shadow-call-stack"

// SCS-R27: "-target-feature" "+scs-reg-r27"
/// -mscs-reg= must not also be swept into a target feature by its option group.
// SCS-R27-NOT: "+scs-reg="

// BAD-REG: error: invalid value 'r15' in '-mscs-reg='
// BAD-REG-28: error: invalid value 'r28' in '-mscs-reg='
// BAD-REG-SP: error: invalid value 'sp' in '-mscs-reg='
