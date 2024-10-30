! RUN: not %flang -### -c --target=i686 -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
! RUN: %flang --target=x86_64 -### -c -mcmodel=tiny %s 2>&1 | FileCheck --check-prefix=TINY %s
! RUN: %flang --target=x86_64 -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
! RUN: %flang --target=x86_64 -### -S -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=KERNEL %s
! RUN: %flang --target=x86_64 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
! RUN: %flang --target=x86_64 -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
! RUN: not %flang -### -c --target=powerpc-linux-gnu -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
! RUN: %flang --target=powerpc-unknown-aix -### -S -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
! RUN: %flang --target=powerpc-unknown-aix -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
! RUN: %flang --target=powerpc-unknown-aix -### -S -mcmodel=medium %s 2> %t.log
! RUN: FileCheck --check-prefix=AIX-MCMEDIUM-OVERRIDE %s < %t.log
! RUN: not %flang -### -c -mcmodel=lager %s 2>&1 | FileCheck --check-prefix=INVALID %s
! RUN: %flang --target=aarch64 -### -S -mcmodel=large -fno-pic %s 2>&1 | FileCheck --check-prefix=LARGE %s
! RUN: %flang --target=aarch64-apple-macosx -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
! RUN: not %flang --target=aarch64 -### -S -mcmodel=large -fpic %s 2>&1 | FileCheck --check-prefix=AARCH64-PIC-LARGE %s
! RUN: not %flang -### -c --target=aarch64 -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
! RUN: not %flang -### -c --target=aarch64 -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=ERR-KERNEL %s
! RUN: not %flang --target=aarch64_32-linux -### -S -mcmodel=small %s 2>&1 | FileCheck --check-prefix=ERR-AARCH64_32 %s
! RUN: %flang --target=loongarch64 -### -S -mcmodel=normal %s 2>&1 | FileCheck --check-prefix=SMALL %s
! RUN: %flang --target=loongarch64 -### -S -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
! RUN: %flang --target=loongarch64 -### -S -mcmodel=extreme %s 2>&1 | FileCheck --check-prefix=LARGE %s
! RUN: not %flang --target=loongarch64 -### -S -mcmodel=tiny %s 2>&1 | FileCheck --check-prefix=ERR-TINY %s
! RUN: not %flang --target=loongarch64 -### -S -mcmodel=small %s 2>&1 | FileCheck --check-prefix=ERR-SMALL %s
! RUN: not %flang --target=loongarch64 -### -S -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=ERR-KERNEL %s
! RUN: not %flang --target=loongarch64 -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=ERR-LARGE %s

! TINY: "-mcmodel=tiny"
! SMALL: "-mcmodel=small"
! KERNEL: "-mcmodel=kernel"
! MEDIUM: "-mcmodel=medium"
! LARGE: "-mcmodel=large"
! AIX-MCMEDIUM-OVERRIDE: "-mcmodel=large"

! INVALID: error: unsupported argument 'lager' to option '-mcmodel=' for target '{{.*}}'

! ERR-TINY:   error: unsupported argument 'tiny' to option '-mcmodel=' for target '{{.*}}'
! ERR-SMALL:  error: unsupported argument 'small' to option '-mcmodel=' for target '{{.*}}'
! ERR-MEDIUM: error: unsupported argument 'medium' to option '-mcmodel=' for target '{{.*}}'
! ERR-KERNEL: error: unsupported argument 'kernel' to option '-mcmodel=' for target '{{.*}}'
! ERR-LARGE:  error: unsupported argument 'large' to option '-mcmodel=' for target '{{.*}}'

! AARCH64-PIC-LARGE: error: invalid argument '-mcmodel=large' only allowed with '-fno-pic'
! ERR-AARCH64_32: error: unsupported argument 'small' to option '-mcmodel=' for target 'aarch64_32-unknown-linux'

