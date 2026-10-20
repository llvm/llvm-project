! UNSUPPORTED: system-windows
! REQUIRES: lld
! check flto-partitions is passed to lld, and not to fc1
! RUN: %flang -### -fuse-ld=lld -flto=full -flto-partitions=16 %s 2>&1 | FileCheck %s --check-prefixes=LLD-PART,FC1-PART

! FC1-PART: "-fc1"
! FC1-PART-SAME: "-flto=full"
! NOT-FC1-PART-SAME: "-flto-partitions=16"
! LLD-PART: ld.lld
! LLD-PART-SAME: "--lto-partitions=16"

! check fat-lto-objects is passed to lld, fc1
! RUN: %flang -### -fuse-ld=lld -flto -ffat-lto-objects %s 2>&1 | FileCheck %s --check-prefixes=LLD-FAT,FC1-FAT

! FC1-FAT: "-fc1"
! FC1-FAT-SAME: "-flto=full"
! FC1-FAT-SAME: "-ffat-lto-objects"
! LLD-FAT: ld.lld
! LLD-FAT-SAME: "--fat-lto-objects"
program test
end program
