// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
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
// CIR-DAG: !rec_Trivial = !cir.record<struct "Trivial" {!s32i, !s32i}>
// CIR-DAG: !rec_Empty = !cir.record<struct "Empty" padded {!u8i}>
// CIR-DAG: !rec_Aligned = !cir.record<struct "Aligned" padded {!s32i, !s32i, !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_NonTrivialDtor = !cir.record<class "NonTrivialDtor" {!s32i}>

// ABI metadata lives in module-level cir.record_layouts attribute.
// CIR-DAG: Trivial = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 4>
// CIR-DAG: Empty = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 1>
// CIR-DAG: Aligned = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 16>
// CIR-DAG: NonTrivialDtor = #cir.record_layout<arg_passing_kind = cannot_pass_in_regs, has_trivial_dtor = false, record_align = 4>
