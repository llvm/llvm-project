// RUN: %clang_cc1 -fno-rtti -fms-extensions -triple aarch64-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-ARM64
// RUN: %clang_cc1 -fno-rtti -fms-extensions -triple arm64ec-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64
// RUN: %clang_cc1 -fno-rtti -fms-extensions -triple x86_64-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

// On Arm64 (but not Arm64EC or x64), MSVC reuses the tail padding of an
// over-aligned base class for a subsequent base class.  This holds whether the
// over-alignment comes from the record itself or from one of its fields.

struct alignas(16) Type {
  long long a, b, c;
};
struct Mid : Type {};
struct Node { void *next; };

// The over-aligned base's tail padding is reused for the following base only on
// Arm64.
struct Derived : Mid, Node {};

int a = sizeof(Derived);

// CHECK-ARM64-LABEL:  0 | struct Derived
// CHECK-ARM64-NEXT:   0 |   struct Mid (base)
// CHECK-ARM64-NEXT:   0 |     struct Type (base)
// CHECK-ARM64:       24 |   struct Node (base)
// CHECK-ARM64-NEXT:  24 |     void * next
// CHECK-ARM64-NEXT:     | [sizeof=32, align=16,
// CHECK-ARM64-NEXT:     |  nvsize=32, nvalign=16]

// CHECK-X64-LABEL:  0 | struct Derived
// CHECK-X64-NEXT:   0 |   struct Mid (base)
// CHECK-X64-NEXT:   0 |     struct Type (base)
// CHECK-X64:       32 |   struct Node (base)
// CHECK-X64-NEXT:  32 |     void * next
// CHECK-X64-NEXT:     | [sizeof=48, align=16,
// CHECK-X64-NEXT:     |  nvsize=48, nvalign=16]

// The over-alignment can also come from `alignas` on an individual field rather
// than from the record itself; the tail padding is reused the same way.
struct FieldOver {
  alignas(16) long long a;
  long long b, c;
};
struct FieldMid : FieldOver {};
struct FieldDerived : FieldMid, Node {};

int b = sizeof(FieldDerived);

// CHECK-ARM64-LABEL:  0 | struct FieldDerived
// CHECK-ARM64-NEXT:   0 |   struct FieldMid (base)
// CHECK-ARM64-NEXT:   0 |     struct FieldOver (base)
// CHECK-ARM64:       24 |   struct Node (base)
// CHECK-ARM64-NEXT:  24 |     void * next
// CHECK-ARM64-NEXT:     | [sizeof=32, align=16,
// CHECK-ARM64-NEXT:     |  nvsize=32, nvalign=16]

// CHECK-X64-LABEL:  0 | struct FieldDerived
// CHECK-X64-NEXT:   0 |   struct FieldMid (base)
// CHECK-X64-NEXT:   0 |     struct FieldOver (base)
// CHECK-X64:       32 |   struct Node (base)
// CHECK-X64-NEXT:  32 |     void * next
// CHECK-X64-NEXT:     | [sizeof=48, align=16,
// CHECK-X64-NEXT:     |  nvsize=48, nvalign=16]

// Reusing tail padding must not affect placement of an over-aligned base that
// follows another base: padding is still inserted so it lands at its required
// alignment.
struct PadBefore : Node, Type {};

int c = sizeof(PadBefore);

// CHECK-ARM64-LABEL:  0 | struct PadBefore
// CHECK-ARM64-NEXT:   0 |   struct Node (base)
// CHECK-ARM64-NEXT:   0 |     void * next
// CHECK-ARM64-NEXT:  16 |   struct Type (base)
// CHECK-ARM64:          | [sizeof=48, align=16,

// CHECK-X64-LABEL:  0 | struct PadBefore
// CHECK-X64-NEXT:   0 |   struct Node (base)
// CHECK-X64-NEXT:   0 |     void * next
// CHECK-X64-NEXT:  16 |   struct Type (base)
// CHECK-X64:          | [sizeof=48, align=16,

// Over-alignment applied directly to a bitfield via __declspec(align) raises
// the record's alignment (rather than its required alignment) and is immune to
// #pragma pack.  The over-aligned base tail-padding reuse above does not apply
// to bitfields, so this is left unchanged by that transformation and is handled
// identically on every target.
#pragma pack(2)
struct BitfieldAlign {
  __declspec(align(16)) long long a : 8;
};
#pragma pack()

int d = sizeof(BitfieldAlign);

// CHECK-ARM64-LABEL:      0 | struct BitfieldAlign
// CHECK-ARM64-NEXT:   0:0-7 |   long long a
// CHECK-ARM64-NEXT:         | [sizeof=8, align=16,
// CHECK-ARM64-NEXT:         |  nvsize=8, nvalign=16]

// CHECK-X64-LABEL:      0 | struct BitfieldAlign
// CHECK-X64-NEXT:   0:0-7 |   long long a
// CHECK-X64-NEXT:         | [sizeof=8, align=16,
// CHECK-X64-NEXT:         |  nvsize=8, nvalign=16]
