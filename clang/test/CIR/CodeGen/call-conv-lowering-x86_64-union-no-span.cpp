// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Empty {};

// The declared alignment stretches a 4-byte member over 16 bytes, and the
// eightbyte narrows to the member because the rest holds nothing.
union OverAligned { int i; } __attribute__((aligned(16)));
void takeOverAligned(OverAligned u) {}
// CIR: cir.func{{.*}} @_Z15takeOverAligned11OverAligned(%arg0: !s32i loc
// LLVM: define{{.*}} void @_Z15takeOverAligned11OverAligned(i32 %{{.+}})

// Three of the four declared bytes hold data, which rounds up to i32.
union ShortStorage { short s; char c[3]; };
void takeShortStorage(ShortStorage u) {}
// CIR: cir.func{{.*}} @_Z16takeShortStorage12ShortStorage(%arg0: !u32i loc
// LLVM: define{{.*}} void @_Z16takeShortStorage12ShortStorage(i32 %{{.+}})

// One byte of data in four declared bytes.  A union of narrow bit-fields has
// the same size and coerces to i32 instead, which take_bit_extent below pins.
union ByteBlobs { unsigned char c, d; } __attribute__((aligned(4)));
void takeByteBlobs(ByteBlobs u) {}
// CIR: cir.func{{.*}} @_Z13takeByteBlobs9ByteBlobs(%arg0: !u8i loc
// LLVM: define{{.*}} void @_Z13takeByteBlobs9ByteBlobs(i8 %{{.+}})

union PadByte { unsigned char c; } __attribute__((aligned(4)));
void takePadByte(PadByte u) {}
// CIR: cir.func{{.*}} @_Z11takePadByte7PadByte(%arg0: !u8i loc
// LLVM: define{{.*}} void @_Z11takePadByte7PadByte(i8 %{{.+}})

// The same union reached as a member, where the enclosing struct is too large
// for registers.
struct WrapsOverAligned { double d; OverAligned u; };
void takeWrapsOverAligned(WrapsOverAligned s) {}
// CIR: cir.func{{.*}} @_Z20takeWrapsOverAligned16WrapsOverAligned(%arg0: !cir.ptr<!rec_WrapsOverAligned> {llvm.align = 16 : i64, llvm.byval = !rec_WrapsOverAligned, llvm.noundef} loc
// LLVM: define{{.*}} void @_Z20takeWrapsOverAligned16WrapsOverAligned(ptr noundef byval(%struct.WrapsOverAligned) align 16 %{{.+}})

// A bit-field access unit of one byte, in a union the alignment stretches to
// eight.
union BitOverAligned { int x : 3; } __attribute__((aligned(8)));
void takeBitOverAligned(BitOverAligned u) {}
// CIR: cir.func{{.*}} @_Z18takeBitOverAligned14BitOverAligned(%arg0: !u64i loc
// LLVM: define{{.*}} void @_Z18takeBitOverAligned14BitOverAligned(i64 %{{.+}})

// A declared type wider than the union's first eightbyte.
union WideDecl { __int128 x : 100; };
void takeWideDecl(WideDecl u) {}
// CIR: cir.func{{.*}} @_Z12takeWideDecl8WideDecl(%arg0: !u64i loc{{.*}}, %arg1: !u64i loc
// LLVM: define{{.*}} void @_Z12takeWideDecl8WideDecl(i64 %{{[^,]+}}, i64 %{{.+}})

// One access unit holding two declarations.
union MultiDecl { char a : 4; int b : 4; };
void takeMultiDecl(MultiDecl u) {}
// CIR: cir.func{{.*}} @_Z13takeMultiDecl9MultiDecl(%arg0: !u32i loc
// LLVM: define{{.*}} void @_Z13takeMultiDecl9MultiDecl(i32 %{{.+}})

// A named unit beside an unnamed one that reaches further.
union NamedPlusUnnamed { int x : 3; long long : 40; };
void takeNamedPlusUnnamed(NamedPlusUnnamed u) {}
// CIR: cir.func{{.*}} @_Z20takeNamedPlusUnnamed16NamedPlusUnnamed(%arg0: !u64i loc
// LLVM: define{{.*}} void @_Z20takeNamedPlusUnnamed16NamedPlusUnnamed(i64 %{{.+}})

union BitUnnamed { int x : 8; long long : 64; };
void takeBitUnnamed(BitUnnamed u) {}
// CIR: cir.func{{.*}} @_Z14takeBitUnnamed10BitUnnamed(%arg0: !u64i loc
// LLVM: define{{.*}} void @_Z14takeBitUnnamed10BitUnnamed(i64 %{{.+}})

union WideBitUnnamed { int x : 24; long long : 64; };
void takeWideBitUnnamed(WideBitUnnamed u) {}
// CIR: cir.func{{.*}} @_Z18takeWideBitUnnamed14WideBitUnnamed(%arg0: !u64i loc
// LLVM: define{{.*}} void @_Z18takeWideBitUnnamed14WideBitUnnamed(i64 %{{.+}})

// An empty member supplies no bytes, so the short is what the eightbyte is
// sized from.
union EmptyNarrow { Empty e; short s; };
void takeEmptyNarrow(EmptyNarrow u) {}
// CIR: cir.func{{.*}} @_Z15takeEmptyNarrow11EmptyNarrow(%arg0: !s16i loc
// LLVM: define{{.*}} void @_Z15takeEmptyNarrow11EmptyNarrow(i16 %{{.+}})

// A union of narrow bit-fields, for contrast with takeByteBlobs above.
union BitExtent { unsigned a : 1; unsigned b : 1; };
void takeBitExtent(BitExtent u) {}
// CIR: cir.func{{.*}} @_Z13takeBitExtent9BitExtent(%arg0: !u32i loc
// LLVM: define{{.*}} void @_Z13takeBitExtent9BitExtent(i32 %{{.+}})

// The array member covers 12 of the union's 16 declared bytes, and the pointer
// member sets the alignment that rounds it up.
union PayloadOrPtr { unsigned Words[3]; void *Ptr; };
PayloadOrPtr byValue(PayloadOrPtr x) { return x; }
// CIR: cir.func{{.*}} @_Z7byValue12PayloadOrPtr(%arg0: !cir.ptr<!void> loc{{.*}}, %arg1: !u64i loc{{.*}}) -> !rec_anon_struct
// LLVM: define{{.*}} { ptr, i64 } @_Z7byValue12PayloadOrPtr(ptr %{{[^,]+}}, i64 %{{.+}})

// The same shape with a member that does cover all 16 bytes, which coerces
// identically.
union PayloadOrPtr16 { unsigned Words[4]; void *Ptr; };
PayloadOrPtr16 byValue16(PayloadOrPtr16 x) { return x; }
// CIR: cir.func{{.*}} @_Z9byValue1614PayloadOrPtr16(%arg0: !cir.ptr<!void> loc{{.*}}, %arg1: !u64i loc{{.*}}) -> !rec_anon_struct
// LLVM: define{{.*}} { ptr, i64 } @_Z9byValue1614PayloadOrPtr16(ptr %{{[^,]+}}, i64 %{{.+}})

// Nine of the union's 16 bytes hold data, so the second eightbyte narrows to
// the single byte there rather than spanning the tail.
union TailByteOrPtr { char Bytes[9]; void *Ptr; };
void takeTailByteOrPtr(TailByteOrPtr x) {}
// CIR: cir.func{{.*}} @_Z17takeTailByteOrPtr13TailByteOrPtr(%arg0: !cir.ptr<!void> loc{{.*}}, %arg1: !u8i loc
// LLVM: define{{.*}} void @_Z17takeTailByteOrPtr13TailByteOrPtr(ptr %{{[^,]+}}, i8 %{{.+}})

struct ErrorInfoBase;
struct UniquePtrLike { ErrorInfoBase *Ptr; };
struct Payload { unsigned A, B, C; };

// A 16-byte union no member covers, under a bit-field unit that pushes the
// record out of registers.
struct ExpectedLike {
  union { Payload TStorage; UniquePtrLike ErrorStorage; };
  bool HasError : 1;
  bool Unchecked : 1;
};
void takeExpectedLike(ExpectedLike);
ExpectedLike returnExpectedLike(unsigned V) {
  ExpectedLike R{};
  R.TStorage.A = V;
  return R;
}
// CIR: cir.func{{.*}} @_Z18returnExpectedLikej(%arg0: !cir.ptr<!rec_ExpectedLike> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_ExpectedLike, llvm.writable}
// LLVM: define{{.*}} void @_Z18returnExpectedLikej(ptr dead_on_unwind noalias writable sret(%struct.ExpectedLike) align 8 %{{[^,]+}}, i32 noundef %{{.+}})

// A 16-byte union with no bit-fields, inside a larger record.
struct Rec;
struct ResOperand {
  enum { RenderAsmOperand, TiedOperand } Kind;
  struct TiedOperandsTuple { unsigned ResOpnd, SrcOpnd1Idx, SrcOpnd2Idx; };
  union {
    unsigned AsmOperandNum;
    TiedOperandsTuple TiedOperands;
    long long ImmVal;
    const Rec *Register;
  };
  unsigned MINumOperands;
};
ResOperand getTiedOp(unsigned Tied) {
  ResOperand X{};
  X.Kind = ResOperand::TiedOperand;
  X.AsmOperandNum = Tied;
  X.MINumOperands = 1;
  return X;
}
// CIR: cir.func{{.*}} @_Z9getTiedOpj(%arg0: !cir.ptr<!rec_ResOperand> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_ResOperand, llvm.writable}
// LLVM: define{{.*}} void @_Z9getTiedOpj(ptr dead_on_unwind noalias writable sret(%struct.ResOperand) align 8 %{{[^,]+}}, i32 noundef %{{.+}})
