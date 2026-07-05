// RUN: %clang_cc1 -ast-dump -Wno-unused %s | FileCheck --strict-whitespace %s

// CHECK: FunctionDecl 0x{{[^ ]*}} <{{.*}}:[[@LINE+1]]:1, line:{{[0-9]*}}:1> line:[[@LINE+1]]:6 func 'void ()'
void func() {
  // Ensure that we calculate the correct type from the literal suffix.

  // Note: 0__wb should create an _BitInt(2) because a signed bit-precise
  // integer requires one bit for the sign and one bit for the value,
  // at a minimum.
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:29> col:29 zero_wb 'typeof (0wb)':'_BitInt(2)'
  typedef __typeof__(0__wb) zero_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:30> col:30 neg_zero_wb 'typeof (-0wb)':'_BitInt(2)'
  typedef __typeof__(-0__wb) neg_zero_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:29> col:29 one_wb 'typeof (1wb)':'_BitInt(2)'
  typedef __typeof__(1__wb) one_wb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:30> col:30 neg_one_wb 'typeof (-1wb)':'_BitInt(2)'
  typedef __typeof__(-1__wb) neg_one_wb;

  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:30> col:30 zero_uwb 'typeof (0uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(0__uwb) zero_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:31> col:31 neg_zero_uwb 'typeof (-0uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(-0__uwb) neg_zero_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:30> col:30 one_uwb 'typeof (1uwb)':'unsigned _BitInt(1)'
  typedef __typeof__(1__uwb) one_uwb;

  // Try a value that is too large to fit in [u]intmax_t.

  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:49> col:49 huge_uwb 'typeof (18446744073709551616uwb)':'unsigned _BitInt(65)'
  typedef __typeof__(18446744073709551616__uwb) huge_uwb;
  // CHECK: TypedefDecl 0x{{[^ ]*}} <col:3, col:48> col:48 huge_wb 'typeof (18446744073709551616wb)':'_BitInt(66)'
  typedef __typeof__(18446744073709551616__wb) huge_wb;
}
