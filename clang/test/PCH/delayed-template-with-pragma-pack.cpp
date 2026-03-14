// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdelayed-template-parsing -emit-pch -o %t.delayed.pch %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -DMAIN_FILE \
// RUN:   -include-pch %t.pch \
// RUN:   -emit-llvm -verify -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -DMAIN_FILE -fdelayed-template-parsing \
// RUN:   -include-pch %t.delayed.pch \
// RUN:   -emit-llvm -verify -o - %s | FileCheck %s

#ifndef MAIN_FILE

extern "C" void consume(int b);

template <int I>
void function() {
#pragma pack(push, 1)
  struct packedAt1 {
    char a;
    unsigned long long b;
    char c;
    unsigned long long d;
    // 18 bytes total
  };
#pragma pack(push, slot1, 2)
  struct packedAt2 {
    char a; // +1 byte of padding
    unsigned long long b;
    char c; // +1 byte of padding
    unsigned long long d;
    // 20 bytes total
  };
#pragma pack(push, 4)
  struct packedAt4 {
    char a; // +3 bytes of padding
    unsigned long long b;
    char c; // +3 bytes of padding
    unsigned long long d;
    // 24 bytes total
  };
#pragma pack(push, 16)
  struct packedAt16 {
    char a; // +7 bytes of padding
    unsigned long long b;
    char c; // +7 bytes of padding
    unsigned long long d;
    // 32 bytes total
  };
#pragma pack(pop, slot1) // This should return packing to 1 (established before push(slot1))
  struct packedAfterPopBackTo1 {
    char a;
    unsigned long long b;
    char c;
    unsigned long long d;
  };
#pragma pack(pop)

  consume(sizeof(packedAt1)); // 18
  consume(sizeof(packedAt2)); // 20
  consume(sizeof(packedAt4)); // 24
  consume(sizeof(packedAt16)); // 32
  consume(sizeof(packedAfterPopBackTo1)); // 18 again
}

#else

// CHECK-LABEL: define linkonce_odr dso_local void @"??$function@$0A@@@YAXXZ"(
// CHECK: call void @consume(i32 noundef 18)
// CHECK-NEXT: call void @consume(i32 noundef 20)
// CHECK-NEXT: call void @consume(i32 noundef 24)
// CHECK-NEXT: call void @consume(i32 noundef 32)
// CHECK-NEXT: call void @consume(i32 noundef 18)
void foo() {
  function<0>();
}

// expected-no-diagnostics

#endif
