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

void takesTrivial(Trivial t) {}
void takesEmpty(Empty e) {}
void takesAligned(Aligned a) {}
void takesNTD(NonTrivialDtor n) {}

// Record types should NOT contain ABI metadata keywords.
// CIR-DAG: !rec_Trivial = !cir.struct<"Trivial" {data !s32i, data !s32i}>
// CIR-DAG: !rec_Empty = !cir.struct<"Empty" {pad !u8i}>
// CIR-DAG: !rec_Aligned = !cir.struct<"Aligned" {data !s32i, data !s32i, pad !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_NonTrivialDtor = !cir.struct<class "NonTrivialDtor" {data !s32i}>

// ABI metadata lives in module-level cir.record_layouts attribute.
// CIR-DAG: Trivial = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 4>
// CIR-DAG: Empty = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1>
// CIR-DAG: Aligned = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 16>
// CIR-DAG: NonTrivialDtor = #cir.record_layout<arg_passing_kind = cannot_pass_in_regs, has_trivial_dtor = false, record_align = 4>
