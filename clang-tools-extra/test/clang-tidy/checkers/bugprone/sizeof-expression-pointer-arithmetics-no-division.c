// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-sizeof-expression.WarnOnOffsetDividedBySizeOf: false \
// RUN:   }}'

typedef __SIZE_TYPE__ size_t;

void situational14(int *Buffer, size_t BufferSize) {
  int *P = &Buffer[0];
  while (P < Buffer + BufferSize / sizeof(*Buffer)) {
    // NO-WARNING: This test opted out of "P +- N */ sizeof(...)" warnings.
    ++P;
  }
}
