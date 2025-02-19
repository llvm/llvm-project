// 1) test on platforms that (do or do not) require runtime relocation

// RUN: %clang --target=x86_64-darwin -fprofile-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=NO_RELOC
// NO_RELOC: "-cc1" {{.*}} "-fprofile-continuous"
// NO_RELOC-NOT: "-mllvm" "-runtime-counter-relocation"

// RUN: %clang --target=powerpc64-ibm-aix -fprofile-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=RELOC
// RUN: %clang --target=x86_64-unknown-fuchsia -fprofile-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=RELOC
// RUN: %clang --target=x86_64-windows-msvc -fprofile-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=RELOC
// RELOC: "-cc1" {{.*}} "-fprofile-continuous" "-mllvm" "-runtime-counter-relocation"

// 2) test -fprofile-continuous with cs-profile-generate and -fprofile-instr-generate

// RUN: %clang --target=powerpc-ibm-aix -fprofile-instr-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=CLANG_PGO
// RUN: %clang --target=powerpc64le-unknown-linux -fprofile-instr-generate= -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=CLANG_PGO
// CLANG_PGO: "-cc1" {{.*}} "-fprofile-continuous" "-mllvm" "-runtime-counter-relocation" "-fprofile-instrument-path=default.profraw"

// RUN: %clang --target=x86_64-unknown-fuchsia -fcs-profile-generate -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=RELOC

// RUN: not %clang -fprofile-continuous -### -c %s 2>&1 | FileCheck %s --check-prefix=ERROR
// ERROR: error: invalid argument '-fprofile-continuous' only allowed with '-fprofile-generate, -fprofile-instr-generate, or -fcs-profile-generate'
void foo(){}
