// RUN: %clang_cc1 -emit-llvm -o - %s

// PR4610
#pragma pack(4)
struct ref {
        struct ref *next;
} refs;
