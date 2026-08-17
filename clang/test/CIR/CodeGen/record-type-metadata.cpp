// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports parameters of an empty or tag class and padded, packed, and
// over-aligned record shapes.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct Trivial { int x, y; };
struct Empty {};
struct __attribute__((aligned(16))) Aligned { int a, b; };

class NonTrivialDtor {
  int val;
public:
  ~NonTrivialDtor();
};

// A record whose only member is an unnamed bit-field carries no data for
// argument passing, and one holding a byte of data lowers to the same members.
struct UnnamedBits { int : 3; };
struct OneByte { unsigned char c; };

// Emptiness follows the base classes.  A C++ empty member counts only under
// [[no_unique_address]], and never through an array of them, however many
// elements it has: only a zero-length array is empty.
struct HasEmptyBase : Empty {};
struct HasDataBase : OneByte {};
struct EmptyMem { Empty e; };
struct NoUniqueOne { [[no_unique_address]] Empty e; };
struct ArrOfEmpty { Empty a[2]; };
struct NoUniqueArr { [[no_unique_address]] Empty a[2]; };
struct ZeroArr { int a[0]; };

// A vtable pointer is data, though it is neither a base nor a field.
struct Poly { virtual void f(); };

// A union carries no data when every member is empty, or when it has none.
union UnnamedBitsUnion { unsigned : 3; };
union NoMembers {};

void takesTrivial(Trivial t) {}
void takesEmpty(Empty e) {}
void takesAligned(Aligned a) {}
void takesNTD(NonTrivialDtor n) {}
void takesUnnamedBits(UnnamedBits u) {}
void takesOneByte(OneByte o) {}
void takesHasEmptyBase(HasEmptyBase d) {}
void takesHasDataBase(HasDataBase d) {}
void takesEmptyMem(EmptyMem e) {}
void takesNoUniqueOne(NoUniqueOne n) {}
void takesArrOfEmpty(ArrOfEmpty a) {}
void takesNoUniqueArr(NoUniqueArr n) {}
void takesZeroArr(ZeroArr z) {}
void takesPoly(Poly *p) {}
void takesUnnamedBitsUnion(UnnamedBitsUnion u) {}
void takesNoMembers(NoMembers n) {}

// Record types should NOT contain ABI metadata keywords.
// CIR-DAG: !rec_Trivial = !cir.struct<"Trivial" {data !s32i, data !s32i}>
// CIR-DAG: !rec_Empty = !cir.struct<"Empty" {pad !u8i}>
// CIR-DAG: !rec_Aligned = !cir.struct<"Aligned" {data !s32i, data !s32i, pad !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_NonTrivialDtor = !cir.struct<class "NonTrivialDtor" {data !s32i}>

// UnnamedBits and OneByte are the same type, so only the metadata separates
// the one that carries data from the one that does not.
// CIR-DAG: !rec_UnnamedBits = !cir.struct<"UnnamedBits" {!u8i}>
// CIR-DAG: !rec_OneByte = !cir.struct<"OneByte" {!u8i}>

// ABI metadata lives in module-level cir.record_layouts attribute.
// CIR-DAG: Trivial = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 4, is_empty = false>
// The leading separator keeps this from matching inside a name that ends in
// "Empty", since every entry prints on one line.
// CIR-DAG: {{[{,] }}Empty = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
// CIR-DAG: Aligned = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 16, is_empty = false>
// CIR-DAG: NonTrivialDtor = #cir.record_layout<arg_passing_kind = cannot_pass_in_regs, has_trivial_dtor = false, record_align = 4, is_empty = false>
// CIR-DAG: UnnamedBits = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
// CIR-DAG: OneByte = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = false>
// CIR-DAG: HasEmptyBase = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
// CIR-DAG: HasDataBase = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = false>
// CIR-DAG: ArrOfEmpty = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = false>
// CIR-DAG: NoUniqueArr = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = false>
// CIR-DAG: ZeroArr = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 4, is_empty = true>
// CIR-DAG: EmptyMem = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = false>
// CIR-DAG: NoUniqueOne = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
// CIR-DAG: Poly = #cir.record_layout<arg_passing_kind = cannot_pass_in_regs, has_trivial_dtor = true, record_align = 8, is_empty = false>
// CIR-DAG: UnnamedBitsUnion = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
// CIR-DAG: NoMembers = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1, is_empty = true>
