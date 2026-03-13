// RUN: %clang_cc1 -emit-llvm-only -triple x86_64 %s

/// Packed struct: different storage unit types should pack without alignment
/// padding between storage units.
struct __attribute__((packed,ms_struct)) P1 {
  short a : 8;
  int b : 30;
};

static int p1[(sizeof(struct P1) == 6) ? 1 : -1];

// Packed struct with char and int fields.
struct [[gnu::ms_struct,gnu::packed]] P2 {
  char a : 4;
  int b : 20;
};

static int p2[(sizeof(struct P2) == 5) ? 1 : -1];
