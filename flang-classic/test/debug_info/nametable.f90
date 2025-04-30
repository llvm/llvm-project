!RUN: %flang %s -gdwarf-5 -c -o - | llvm-readelf -S - | FileCheck %s --check-prefix=NONAMESECTION
!RUN: %flang %s -g -c -o - | llvm-readelf -S - | FileCheck %s --check-prefix=NOPUBNAMESECTION

!RUN: %flang %s -gdwarf-5 -gpubnames -c -o - | llvm-readelf -S - | FileCheck %s --check-prefix=NAMESECTION
!RUN: %flang %s -gdwarf-4 -gpubnames -c -o - | llvm-readelf -S - | FileCheck %s --check-prefix=PUBNAMESECTION

!Ensure that `.debug_names` or `.debug_pubnames` are NOT present by default.
!NONAMESECTION-NOT: .debug_names
!NOPUBNAMESECTION-NOT: .debug_pubnames
!Ensure that `.debug_names` or `.debug_pubnames` are present when `-gpubnames` is specified.
!NAMESECTION: .debug_names
!PUBNAMESECTION: .debug_pubnames

PROGRAM main
END PROGRAM main

