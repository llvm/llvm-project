// RUN: %clang_cc1 -emit-llvm -triple s390x-ibm-zos -fzos-extensions -fdump-record-layouts < %s | FileCheck %s --check-prefix=PTR32-ZOS
// RUN: %clang_cc1 -emit-llvm -triple s390x-ibm-linux -fzos-extensions -fdump-record-layouts < %s | FileCheck %s --check-prefix=PTR32-LINUX
// RUN: %clang_cc1 -emit-llvm -triple s390x-linux-gnu -fzos-extensions -fdump-record-layouts < %s | FileCheck %s --check-prefix=PTR32-LINUX

// PTR32-ZOS:          0  | struct s1
// PTR32-ZOS-NEXT:     0  | long a
// PTR32-ZOS-NEXT:     8  | int b
// PTR32-ZOS-NEXT:     12 | int * __ptr32 c
// PTR32-ZOS-NEXT:     16 | int d
// PTR32-ZOS-NEXT:        | [sizeof=24, align=8]

// PTR32-LINUX:        0  | struct s1
// PTR32-LINUX-NEXT:   0  | long a
// PTR32-LINUX-NEXT:   8  | int b
// PTR32-LINUX-NEXT:   16 | int * __ptr32 c
// PTR32-LINUX-NEXT:   24 | int d
// PTR32-LINUX-NEXT:      | [sizeof=32, align=8]
struct s1 {
  long a;
  int b;
  int * __ptr32 c;
  int d;
} S1;

// PTR32-ZOS:          0  | struct s2
// PTR32-ZOS-NEXT:     0  | long a
// PTR32-ZOS-NEXT:     8  | int b
// PTR32-ZOS-NEXT:     16 | int * c
// PTR32-ZOS-NEXT:     24 | int d
// PTR32-ZOS-NEXT:        | [sizeof=32, align=8]

// PTR32-LINUX:        0  | struct s2
// PTR32-LINUX-NEXT:   0  | long a
// PTR32-LINUX-NEXT:   8  | int b
// PTR32-LINUX-NEXT:   16 | int * c
// PTR32-LINUX-NEXT:   24 | int d
// PTR32-LINUX-NEXT:      | [sizeof=32, align=8]
struct s2 {
  long a;
  int b;
  int *c;
  int d;
} S2;

// PTR32-ZOS:          0  | struct s3
// PTR32-ZOS-NEXT:     0  | int a
// PTR32-ZOS-NEXT:     4  | int * __ptr32 b
// PTR32-ZOS-NEXT:     8  | int * __ptr32 c
// PTR32-ZOS-NEXT:     12 | int * d
// PTR32-ZOS-NEXT:        | [sizeof=20, align=1]

struct __attribute__((packed)) s3 {
  int a;
  int *__ptr32 b;
  int *__ptr32 c;
  int *d;
};
struct s3 S3;

// PTR32-ZOS:          0 | union u1
// PTR32-ZOS-NEXT:     0 | int * __ptr32 a
// PTR32-ZOS-NEXT:     0 | int * b
// PTR32-ZOS-NEXT:       | [sizeof=8, align=8]

// PTR32-LINUX:        0 | union u1
// PTR32-LINUX-NEXT:   0 | int * __ptr32 a
// PTR32-LINUX-NEXT:   0 | int * b
// PTR32-LINUX-NEXT:     | [sizeof=8, align=8]
union u1 {
  int *__ptr32 a;
  int *b;
} U1;

// PTR32-ZOS:          0 | union u2
// PTR32-ZOS-NEXT:     0 | int * __ptr32 a
// PTR32-ZOS-NEXT:     0 | int * b
// PTR32-ZOS-NEXT:       | [sizeof=8, align=1]

union __attribute__((packed)) u2 {
  int *__ptr32 a;
  int *b;
};
union u2 U2;

// PTR32-ZOS:          0 | union u3
// PTR32-ZOS-NEXT:     0 | int * __ptr32 a
// PTR32-ZOS-NEXT:     0 | short b
// PTR32-ZOS-NEXT:       | [sizeof=4, align=1]

union __attribute__((packed)) u3 {
  int *__ptr32 a;
  short b;
};
union u3 U3;
