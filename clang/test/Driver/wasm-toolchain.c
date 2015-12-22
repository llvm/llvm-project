// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown -x assembler %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// AS_LINK: clang{{.*}}" "-cc1as" {{.*}} "-o" "[[temp:[^"]*]]"
// AS_LINK: lld" "-flavor" "ld" "[[temp]]" "-o" "a.out"
