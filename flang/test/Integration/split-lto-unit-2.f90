! Check that -flto=thin without -fsplit-lto-unit has EnableSplitLTOUnit = 0

! UNSUPPORTED: system-darwin

! RUN: %flang -flto=thin  -S -o - %s |  FileCheck %s --check-prefix=SPLIT0
! RUN: %if x86-registered-target %{ %flang -flto=thin --target=x86_64-linux-gnu -S -o - %s |  FileCheck %s --check-prefix=SPLIT0 %}
! RUN: %if x86-registered-target %{ %flang -flto=thin --target=x86_64-apple-macosx -S -o - %s | FileCheck %s --check-prefix=SPLIT0 %}

! Check that -flto=thin with -fsplit-lto-unit has EnableSplitLTOUnit = 1
! RUN: %flang -flto=thin -fsplit-lto-unit -S -o - %s | FileCheck %s --check-prefix=SPLIT1
! RUN: %if x86-registered-target %{ %flang -flto=thin --target=x86_64-linux-gnu -fsplit-lto-unit -S -o - %s | FileCheck %s --check-prefix=SPLIT1 %}
! RUN: %if x86-registered-target %{ %flang -flto=thin --target=x86_64-apple-macosx -fsplit-lto-unit -S -o - %s | FileCheck %s --check-prefix=SPLIT1 %}

! Check that regular LTO has EnableSplitLTOUnit = 1 
! RUN: %flang -flto -S -o - %s | FileCheck %s --implicit-check-not="EnableSplitLTOUnit" --check-prefix=SPLIT1
! RUN: %if x86-registered-target %{ %flang -flto --target=x86_64-linux-gnu -S -o - %s |  FileCheck %s --check-prefix=SPLIT1 %}

! Check that regular LTO has no EnableSplitLTOUnit for apple targets
! RUN: %if x86-registered-target %{ %flang -flto --target=x86_64-apple-macosx -S -o - %s | FileCheck %s --check-prefix=NOSPLIT %}

! SPLIT0: !{i32 1, !"EnableSplitLTOUnit", i32 0}
! SPLIT1: !{i32 1, !"EnableSplitLTOUnit", i32 1}
! NOSPLIT-NOT: "EnableSplitLTOUnit"

program main
end program main
