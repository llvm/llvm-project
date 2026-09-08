// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -emit-obj %s -o - | llvm-readobj -h - | FileCheck %s --check-prefix=GNU-DEFAULT
// GNU-DEFAULT: TimeDateStamp: 1970-01-01 00:00:00 (0x0)

// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -mincremental-linker-compatible -emit-obj %s -o - | llvm-readobj -h - | FileCheck %s --check-prefix=GNU-INC
// GNU-INC: ImageFileHeader {
// GNU-INC:   TimeDateStamp:
// GNU-INC-NOT: 1970-01-01 00:00:00 (0x0)

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -mno-incremental-linker-compatible -emit-obj %s -o - | llvm-readobj -h - | FileCheck %s --check-prefix=MSVC-NOINC
// MSVC-NOINC: TimeDateStamp: 1970-01-01 00:00:00 (0x0)

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-obj %s -o - | llvm-readobj -h - | FileCheck %s --check-prefix=MSVC-DEFAULT
// MSVC-DEFAULT: ImageFileHeader {
// MSVC-DEFAULT:   TimeDateStamp:
// MSVC-DEFAULT-NOT: 1970-01-01 00:00:00 (0x0)


int main(void) {
  return 0;
}
