// RUN: %clang_cc1 -std=c2x -ast-dump -Wno-unused %s | FileCheck --strict-whitespace %s

// CHECK: FunctionDecl 0x{{[^ ]*}} <{{.*}}:[[@LINE+1]]:1, line:{{[0-9]*}}:1> line:[[@LINE+1]]:6 func 'void (void)'
void func(void) {
  // Ensure that we calculate the correct type from the literal suffix.

  // Note: 0wb should create an _BitInt(2) because a signed bit-precise
  // integer requires one bit for the sign and one bit for the value,
  // at a minimum.
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:27> col:27 zero_wb 'typeof (0wb)':'_BitInt(2)'
  typedef __typeof__(0wb) zero_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 neg_zero_wb 'typeof (-0wb)':'_BitInt(2)'
  typedef __typeof__(-0wb) neg_zero_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:27> col:27 one_wb 'typeof (1wb)':'_BitInt(2)'
  typedef __typeof__(1wb) one_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 neg_one_wb 'typeof (-1wb)':'_BitInt(2)'
  typedef __typeof__(-1wb) neg_one_wb;

  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 zero_uwb 'typeof (0uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(0uwb) zero_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:29> col:29 neg_zero_uwb 'typeof (-0uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(-0uwb) neg_zero_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 one_uwb 'typeof (1uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(1uwb) one_uwb;

  // Try a value that is too large to fit in [u]intmax_t.

  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:47> col:47 huge_uwb 'typeof (18446744073709551616uwb)':'unsigned _BitInt(65)'
  typedef __typeof__(18446744073709551616uwb) huge_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:46> col:46 huge_wb 'typeof (18446744073709551616wb)':'_BitInt(66)'
  typedef __typeof__(18446744073709551616wb) huge_wb;
}

// Test the examples from the paper.
// CHECK: FunctionDecl 0x{{[^ ]*}} <{{.*}}:[[@LINE+1]]:1, line:{{[0-9]*}}:1> line:[[@LINE+1]]:6 from_paper 'void (void)'
void from_paper(void) {
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 neg_three_wb 'typeof (-3wb)':'_BitInt(3)'
  typedef __typeof__(-3wb) neg_three_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:30> col:30 neg_three_hex_wb 'typeof (-3wb)':'_BitInt(3)'
  typedef __typeof__(-0x3wb) neg_three_hex_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:27> col:27 three_wb 'typeof (3wb)':'_BitInt(3)'
  typedef __typeof__(3wb) three_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:28> col:28 three_uwb 'typeof (3uwb)':'unsigned _BitInt(2)'
  typedef __typeof__(3uwb) three_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:29> col:29 neg_three_uwb 'typeof (-3uwb)':'unsigned _BitInt(2)'
  typedef __typeof__(-3uwb) neg_three_uwb;
}
