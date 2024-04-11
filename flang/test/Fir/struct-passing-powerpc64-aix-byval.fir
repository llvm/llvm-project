// Test powerpc64 rewrite of struct passed by value (BIND(C), VALUE derived types).
//
// RUN: fir-opt --target-rewrite="target=powerpc64-ibm-aix" %s | FileCheck %s
// REQUIRES: powerpc-registered-target

// character type
func.func @csubch(%arg0: !fir.type<_QFcsubchTdt<{c:!fir.char<1>}>> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubch"} { return }
//CHECK-LABEL: func.func @csubch(%arg0: !fir.ref<!fir.type<_QFcsubchTdt<{c:!fir.char<1>}>>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubchTdt<{c:!fir.char<1>}>>}) attributes {fir.bindc_name = "csubch"}

// integer type
func.func @csubi1(%arg0: !fir.type<_QFcsubi1Tdt<{i:i32}>> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubi1"} { return }
//CHECK-LABEL: func.func @csubi1(%arg0: !fir.ref<!fir.type<_QFcsubi1Tdt<{i:i32}>>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubi1Tdt<{i:i32}>>}) attributes {fir.bindc_name = "csubi1"}

// real type (scalar)
func.func @csubr1(%arg0: !fir.type<_QFcsubr1Tdt<{r1:f32,r2:f32,r3:f32}>> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr1"} { return }
//CHECK-LABEL: func.func @csubr1(%arg0: !fir.ref<!fir.type<_QFcsubr1Tdt<{r1:f32,r2:f32,r3:f32}>>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubr1Tdt<{r1:f32,r2:f32,r3:f32}>>}) attributes {fir.bindc_name = "csubr1"}

// real type (array)
func.func @csubr5(%arg0: !fir.type<_QFcsubr5Tdt<{r:!fir.array<8xf32>}>> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr5"} { return }
//CHECK-LABEL: func.func @csubr5(%arg0: !fir.ref<!fir.type<_QFcsubr5Tdt<{r:!fir.array<8xf32>}>>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubr5Tdt<{r:!fir.array<8xf32>}>>}) attributes {fir.bindc_name = "csubr5"}

// mixed types
func.func @csub1(%arg0: !fir.type<_QFcsub1Tdt<{c:!fir.char<1>,r:f32,i:i64}>> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub1"} { return }
//CHECK-LABEL: func.func @csub1(%arg0: !fir.ref<!fir.type<_QFcsub1Tdt<{c:!fir.char<1>,r:f32,i:i64}>>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsub1Tdt<{c:!fir.char<1>,r:f32,i:i64}>>}) attributes {fir.bindc_name = "csub1"}

