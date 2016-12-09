// RUN: %clang -emit-llvm -O0 -S -g %s -o - | FileCheck %s

// Per PR26619, check that for referenced static const of floating-point type,
// we emit its constant value in debug info.  NOTE that PR26619 is not yet fixed for long double.

static const __fp16 hVal = 29/13.0f;            //    2.2307692307692307692     (2.23046875)

static const float fVal = -147/17.0f;           //   -8.6470588235294117647     (-8.64705849)

static const double dVal = 19637/7.0;           // 2805.2857142857142857        (2805.2857142857142)

static const long double ldVal = 3/1234567.0L;  //    2.4300017739012949479e-06 (<optimized out>)

int main() {
  return hVal + fVal + dVal + ldVal;
}

// CHECK: !DIGlobalVariable(name: "hVal", {{.*}}, isLocal: true, isDefinition: true, expr: ![[HEXPR:[0-9]+]]
// CHECK: ![[HEXPR]] = !DIExpression(DW_OP_constu, 16502, DW_OP_stack_value)

// CHECK: !DIGlobalVariable(name: "fVal", {{.*}}, isLocal: true, isDefinition: true, expr: ![[FEXPR:[0-9]+]]
// CHECK: ![[FEXPR]] = !DIExpression(DW_OP_constu, 3238681178, DW_OP_stack_value)

// CHECK: !DIGlobalVariable(name: "dVal", {{.*}}, isLocal: true, isDefinition: true, expr: ![[DEXPR:[0-9]+]]
// CHECK: ![[DEXPR]] = !DIExpression(DW_OP_constu, 4658387303597904457, DW_OP_stack_value)

// Temporarily removing this check -- for some targets (such as
// "--target=hexagon-unknown-elf"), long double does not exceed 64
// bits, and so we actually do get the constant value (expr) emitted.
//
// DO-NOT-CHECK: !DIGlobalVariable(name: "ldVal", {{.*}}, isLocal: true, isDefinition: true)
