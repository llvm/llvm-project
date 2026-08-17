! Test that flang forwards -fno-omit-frame-pointer and -fomit-frame-pointer Flang frontend
! RUN: %flang --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NOVALUE
! CHECK-NOVALUE: "-fc1"{{.*}}"-mframe-pointer=non-leaf-no-reserve"

! RUN: %flang -fomit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONEFP
! CHECK-NONEFP: "-fc1"{{.*}}"-mframe-pointer=none"

! RUN: %flang -fno-omit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONLEAFFP
! CHECK-NONLEAFFP: "-fc1"{{.*}}"-mframe-pointer=non-leaf-no-reserve"

! RUN: %flang -fno-omit-frame-pointer --target=x86-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ALLFP
! CHECK-ALLFP: "-fc1"{{.*}}"-mframe-pointer=all"

! RUN: %flang -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-FRAME-POINTER-ALL
! CHECK-FRAME-POINTER-ALL: "-fc1"{{.*}}"-mframe-pointer=all"

! RUN: %flang -fno-omit-frame-pointer -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-X86-NONLEAF
! CHECK-X86-NONLEAF: "-fc1"{{.*}}"-mframe-pointer=non-leaf-no-reserve"

! RUN: %flang -fno-omit-frame-pointer -momit-leaf-frame-pointer -mno-omit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-LAST-WINS
! CHECK-LAST-WINS: "-fc1"{{.*}}"-mframe-pointer=all"

! Without -fno-omit-frame-pointer the leaf option is silently allowed but has no effect, matching Clang's behavior.
! RUN: %flang -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -O2 -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-LEAF-ONLY
! CHECK-LEAF-ONLY: "-fc1"{{.*}}"-mframe-pointer=none"

subroutine test
end subroutine test
