class ImmCheckType<int val> {
  int Value = val;
}

// These must be kept in sync with the flags in include/clang/Basic/TargetBuiltins.h
def ImmCheck0_31                : ImmCheckType<0>;  // 0..31 (used for e.g. predicate patterns)
def ImmCheck1_16                : ImmCheckType<1>;  // 1..16
def ImmCheckExtract             : ImmCheckType<2>;  // 0..(2048/sizeinbits(elt) - 1)
def ImmCheckShiftRight          : ImmCheckType<3>;  // 1..sizeinbits(elt)
def ImmCheckShiftRightNarrow    : ImmCheckType<4>;  // 1..sizeinbits(elt)/2
def ImmCheckShiftLeft           : ImmCheckType<5>;  // 0..(sizeinbits(elt) - 1)
def ImmCheck0_7                 : ImmCheckType<6>;  // 0..7
def ImmCheckLaneIndex           : ImmCheckType<7>;  // 0..(sizeinbits(vec)/(sizeinbits(elt)) - 1)
def ImmCheckCvt                 : ImmCheckType<8>;  // 1..sizeinbits(elt) (same as ShiftRight)
def ImmCheckLaneIndexCompRotate : ImmCheckType<9>;  // 0..(sizeinbits(vec)/(2*sizeinbits(elt)) - 1)
def ImmCheckLaneIndexDot        : ImmCheckType<10>; // 0..(sizeinbits(vec)/(4*sizeinbits(elt)) - 1)
def ImmCheckComplexRot90_270    : ImmCheckType<11>; // [90,270]
def ImmCheckComplexRotAll90     : ImmCheckType<12>; // [0, 90, 180,270]
def ImmCheck0_13                : ImmCheckType<13>; // 0..13
def ImmCheck0_1                 : ImmCheckType<14>; // 0..1
def ImmCheck0_2                 : ImmCheckType<15>; // 0..2
def ImmCheck0_3                 : ImmCheckType<16>; // 0..3
def ImmCheck0_0                 : ImmCheckType<17>; // 0..0
def ImmCheck0_15                : ImmCheckType<18>; // 0..15
def ImmCheck0_255               : ImmCheckType<19>; // 0..255
def ImmCheck2_4_Mul2            : ImmCheckType<20>; // 2, 4
def ImmCheck1_1                 : ImmCheckType<21>; // 1..1
def ImmCheck1_3                 : ImmCheckType<22>; // 1..3
def ImmCheck1_7                 : ImmCheckType<23>; // 1..7
def ImmCheck1_32                : ImmCheckType<24>; // 1..32
def ImmCheck1_64                : ImmCheckType<25>; // 1..64
def ImmCheck0_63                : ImmCheckType<26>; // 0..63

class ImmCheck<int immArgIdx, ImmCheckType kind, int typeArgIdx = -1> {
  // Parameter index of immediate argument to be verified
  int ImmArgIdx = immArgIdx;

  // Parameter index of argument whose type determines the context of this immediate check -
  // element type for SVE/SME, element type and vector size for NEON (ignoring element type for
  // ClassB NEON intrinsics).
  int TypeContextArgIdx = typeArgIdx;
  ImmCheckType Kind = kind;
}
