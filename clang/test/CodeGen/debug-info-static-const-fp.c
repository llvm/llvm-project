// RUN: %clang_cc1 -emit-llvm -O0 -debug-info-kind=limited %s -o - | \
// RUN:   FileCheck --check-prefixes CHECK %s

// RUN: %clang_cc1 -triple hexagon-unknown--elf -emit-llvm -O0 -debug-info-kind=limited %s -o - | \
// RUN:   FileCheck --check-prefixes CHECK,CHECK-LDsm %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -debug-info-kind=limited %s -o - | \
// RUN:   FileCheck --check-prefixes CHECK,CHECK-LDlg %s

// Per PR26619, check that for referenced static const of floating-point type,
// we emit its constant value in debug info.
//
// NOTE that __fp16 is assumed to be 16 bits, float is assumed to be
// 32 bits, and double is assumed to be 64 bits.  Size of long double
// is not known (for example, it is 64 bits for hexagon-unknown--elf,
// but 128 bits for x86_64-unknown-linux-gnu).  Therefore, we specify
// target triples where it has a known size, and check accordingly:
// for the absence of a constant (CHECK-LDlg) when the size exceeds 64
// bits, and for the presence of a constant (CHECK-LDsm) but not its
// value when the size does not exceed 64 bits.
//
// NOTE that PR26619 is not yet fixed for types greater than 64 bits.

static const __fp16 hVal = 29/13.0f;            //    2.2307692307692307692     (2.23046875)

static const float fVal = -147/17.0f;           //   -8.6470588235294117647     (-8.64705849)

static const double dVal = 19637/7.0;           // 2805.2857142857142857        (2805.2857142857142)

static const long double ldVal = 3/1234567.0L;  //    2.4300017739012949479e-06 (<depends on size of long double>)

int main() {
  return hVal + fVal + dVal + ldVal;
}

// CHECK: !DIGlobalVariableExpression(var: [[HVAL:.*]], expr: !DIExpression(DW_OP_constu, 16502, DW_OP_stack_value))
// CHECK: [[HVAL]] = distinct !DIGlobalVariable(name: "hVal",
// CHECK-SAME:                                  isLocal: true, isDefinition: true

// CHECK: !DIGlobalVariableExpression(var: [[FVAL:.*]], expr: !DIExpression(DW_OP_constu, 3238681178, DW_OP_stack_value))
// CHECK: [[FVAL]] = distinct !DIGlobalVariable(name: "fVal",
// CHECK-SAME:                                  isLocal: true, isDefinition: true

// CHECK: !DIGlobalVariableExpression(var: [[DVAL:.*]], expr: !DIExpression(DW_OP_constu, 4658387303597904457, DW_OP_stack_value))
// CHECK: [[DVAL]] = distinct !DIGlobalVariable(name: "dVal",
// CHECK-SAME:                                  isLocal: true, isDefinition: true

// CHECK-LDlg-DAG: [[LDVAL:.*]] = distinct !DIGlobalVariable(name: "ldVal", {{.*}}, isLocal: true, isDefinition: true)
// CHECK-LDlg-DAG: !DIGlobalVariableExpression(var: [[LDVAL]])
// CHECK-LDsm-DAG: [[LDVAL:.*]] = distinct !DIGlobalVariable(name: "ldVal", {{.*}}, isLocal: true, isDefinition: true)
// CHECK-LDsm-DAG: !DIGlobalVariableExpression(var: [[LDVAL]], expr:
