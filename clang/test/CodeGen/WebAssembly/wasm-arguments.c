// RUN: %clang_cc1 -triple wasm32-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY64
// RUN: %clang_cc1 -triple wasm32-unknown-unknown %s -target-abi experimental-mv -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=EXPERIMENTAL-MV

// Basic argument/attribute and return tests for WebAssembly

// WEBASSEMBLY32: define void @misc_args(i32 noundef %i, i32 noundef %j, i64 noundef %k, double noundef %l, fp128 noundef %m)
// WEBASSEMBLY64: define void @misc_args(i32 noundef %i, i64 noundef %j, i64 noundef %k, double noundef %l, fp128 noundef %m)
void misc_args(int i, long j, long long k, double l, long double m) {}

typedef struct {
  int aa;
  int bb;
} s1;

// Structs should be passed byval and not split up.
// WEBASSEMBLY32: define void @struct_arg(ptr noundef byval(%struct.s1) align 4 %i)
// WEBASSEMBLY64: define void @struct_arg(ptr noundef byval(%struct.s1) align 4 %i)

// Except in the experimental multivalue ABI, where structs are passed in args
// EXPERIMENTAL-MV: define void @struct_arg(i32 %i.0, i32 %i.1)
void struct_arg(s1 i) {}

// Structs should be returned sret and not simplified by the frontend.
// WEBASSEMBLY32: define void @struct_ret(ptr dead_on_unwind noalias writable sret(%struct.s1) align 4 %agg.result)
// WEBASSEMBLY32: ret void
// WEBASSEMBLY64: define void @struct_ret(ptr dead_on_unwind noalias writable sret(%struct.s1) align 4 %agg.result)
// WEBASSEMBLY64: ret void

// Except with the experimental multivalue ABI, which returns structs by value
// EXPERIMENTAL-MV: define %struct.s1 @struct_ret()
// EXPERIMENTAL-MV: ret %struct.s1 %0
s1 struct_ret(void) {
  s1 foo;
  return foo;
}

typedef struct {
  int cc;
} s2;

// Single-element structs should be passed as the one element.
// WEBASSEMBLY32: define void @single_elem_arg(i32 %i.coerce)
// WEBASSEMBLY64: define void @single_elem_arg(i32 %i.coerce)
// EXPERIMENTAL-MV: define void @single_elem_arg(i32 %i.coerce)
void single_elem_arg(s2 i) {}

// Single-element structs should be passed as the one element.
// WEBASSEMBLY32: define i32 @single_elem_ret()
// WEBASSEMBLY32: ret i32
// WEBASSEMBLY64: define i32 @single_elem_ret()
// EXPERIMENTAL-MV: define i32 @single_elem_ret()
s2 single_elem_ret(void) {
  s2 foo;
  return foo;
}

// WEBASSEMBLY32: define void @long_long_arg(i64 noundef %i)
// WEBASSEMBLY64: define void @long_long_arg(i64 noundef %i)
void long_long_arg(long long i) {}

// i8/i16 should be signext, i32 and higher should not.
// WEBASSEMBLY32: define void @char_short_arg(i8 noundef signext %a, i16 noundef signext %b)
// WEBASSEMBLY64: define void @char_short_arg(i8 noundef signext %a, i16 noundef signext %b)
void char_short_arg(char a, short b) {}

// WEBASSEMBLY32: define void @uchar_ushort_arg(i8 noundef zeroext %a, i16 noundef zeroext %b)
// WEBASSEMBLY64: define void @uchar_ushort_arg(i8 noundef zeroext %a, i16 noundef zeroext %b)
void uchar_ushort_arg(unsigned char a, unsigned short b) {}

enum my_enum {
  ENUM1,
  ENUM2,
  ENUM3,
};

// Enums should be treated as the underlying i32.
// WEBASSEMBLY32: define void @enum_arg(i32 noundef %a)
// WEBASSEMBLY64: define void @enum_arg(i32 noundef %a)
void enum_arg(enum my_enum a) {}

enum my_big_enum {
  ENUM4 = 0xFFFFFFFFFFFFFFFF,
};

// Big enums should be treated as the underlying i64.
// WEBASSEMBLY32: define void @big_enum_arg(i64 noundef %a)
// WEBASSEMBLY64: define void @big_enum_arg(i64 noundef %a)
void big_enum_arg(enum my_big_enum a) {}

union simple_union {
  int a;
  char b;
};

// Unions should be passed as byval structs.
// WEBASSEMBLY32: define void @union_arg(ptr noundef byval(%union.simple_union) align 4 %s)
// WEBASSEMBLY64: define void @union_arg(ptr noundef byval(%union.simple_union) align 4 %s)
// EXPERIMENTAL-MV: define void @union_arg(i32 %s.0)
void union_arg(union simple_union s) {}

// Unions should be returned sret and not simplified by the frontend.
// WEBASSEMBLY32: define void @union_ret(ptr dead_on_unwind noalias writable sret(%union.simple_union) align 4 %agg.result)
// WEBASSEMBLY32: ret void
// WEBASSEMBLY64: define void @union_ret(ptr dead_on_unwind noalias writable sret(%union.simple_union) align 4 %agg.result)
// WEBASSEMBLY64: ret void

// The experimental multivalue ABI returns them by value, though.
// EXPERIMENTAL-MV: define %union.simple_union @union_ret()
// EXPERIMENTAL-MV: ret %union.simple_union %0
union simple_union union_ret(void) {
  union simple_union bar;
  return bar;
}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;

// Bitfields should be passed as byval structs.
// WEBASSEMBLY32: define void @bitfield_arg(ptr noundef byval(%struct.bitfield1) align 4 %bf1)
// WEBASSEMBLY64: define void @bitfield_arg(ptr noundef byval(%struct.bitfield1) align 4 %bf1)
// EXPERIMENTAL-MV: define void @bitfield_arg(ptr noundef byval(%struct.bitfield1) align 4 %bf1)
void bitfield_arg(bitfield1 bf1) {}

// And returned via sret pointers.
// WEBASSEMBLY32: define void @bitfield_ret(ptr dead_on_unwind noalias writable sret(%struct.bitfield1) align 4 %agg.result)
// WEBASSEMBLY64: define void @bitfield_ret(ptr dead_on_unwind noalias writable sret(%struct.bitfield1) align 4 %agg.result)

// Except, of course, in the experimental multivalue ABI
// EXPERIMENTAL-MV: define %struct.bitfield1 @bitfield_ret()
bitfield1 bitfield_ret(void) {
  bitfield1 baz;
  return baz;
}
