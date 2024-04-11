// Test ppc64le rewrite of struct passed by value (BIND(C), VALUE derived types).
//
// RUN: fir-opt --target-rewrite="target=ppc64le-unknown-linux" %s | FileCheck %s
// REQUIRES: powerpc-registered-target

// character type
func.func @csubch(%arg0: !fir.type<_QFcsubchTdt{c:!fir.char<1>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubch"} { return }
//CHECK-LABEL: func.func @csubch(%arg0: !fir.array<1xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubch"}

// integer type
func.func @csubi1(%arg0: !fir.type<_QFcsubi1Tdt{i:i32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubi1"} { return }
//CHECK-LABEL: func.func @csubi1(%arg0: !fir.array<1xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubi1"}

// integer type with size > 64 bytes
func.func @csubi2(%arg0: !fir.type<_QFcsubi2Tdt{i:!fir.array<17xi32>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubi2"} { return }
//CHECK-LABEL: func.func @csubi2(%arg0: !fir.ref<!fir.type<_QFcsubi2Tdt{i:!fir.array<17xi32>}>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubi2Tdt{i:!fir.array<17xi32>}>}) attributes {fir.bindc_name = "csubi2"}

// real type (scalar)
func.func @csubr1(%arg0: !fir.type<_QFcsubr1Tdt{r1:f32,r2:f32,r3:f32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr1"} { return }
//CHECK-LABEL: func.func @csubr1(%arg0: !fir.array<3xf32> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr1"}

// real type (< 8 floats and <= 64 bytes)
func.func @csubr2(%arg0: !fir.type<_QFcsubr2Tdt{r1:f64,r2:f64,r3:f64,r4:f64,r5:f64,r6:f64,r7:f64,r8:f64}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr2"} { return }
//CHECK-LABEL: func.func @csubr2(%arg0: !fir.array<8xf64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr2"}

// real type (> 8 floats and <= 64 bytes)
func.func @csubr3(%arg0: !fir.type<_QFcsubr3Tdt{r1:f32,r2:f32,r3:f32,r4:f32,r5:f32,r6:f32,r7:f32,r8:f32,r9:f32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr3"} { return }
//CHECK-LABEL: func.func @csubr3(%arg0: !fir.array<5xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr3"}

// real type (> 8 floats and > 64 bytes)
func.func @csubr4(%arg0: !fir.type<_QFcsubr4Tdt{r1:f32,r2:f32,r3:f32,r4:f32,r5:f32,r6:f32,r7:f32,r8:f32,r9:f32,r10:f32,r11:f32,r12:f32,r13:f32,r14:f32,r15:f32,r16:f32,r17:f32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr4"} { return }
//CHECK-LABEL: func.func @csubr4(%arg0: !fir.ref<!fir.type<_QFcsubr4Tdt{r1:f32,r2:f32,r3:f32,r4:f32,r5:f32,r6:f32,r7:f32,r8:f32,r9:f32,r10:f32,r11:f32,r12:f32,r13:f32,r14:f32,r15:f32,r16:f32,r17:f32}>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubr4Tdt{r1:f32,r2:f32,r3:f32,r4:f32,r5:f32,r6:f32,r7:f32,r8:f32,r9:f32,r10:f32,r11:f32,r12:f32,r13:f32,r14:f32,r15:f32,r16:f32,r17:f32}>}) attributes {fir.bindc_name = "csubr4"}

// real type (array)
func.func @csubr5(%arg0: !fir.type<_QFcsubr5Tdt{r:!fir.array<8xf32>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr5"} { return }
//CHECK-LABEL: func.func @csubr5(%arg0: !fir.array<8xf32> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr5"}

// real type (array componets and > 64 bytes)
func.func @csubr6(%arg0: !fir.type<_QFcsubr6Tdt{r:!fir.array<9xf64>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr6"} { return }
//CHECK-LABEL: func.func @csubr6(%arg0: !fir.ref<!fir.type<_QFcsubr6Tdt{r:!fir.array<9xf64>}>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubr6Tdt{r:!fir.array<9xf64>}>}) attributes {fir.bindc_name = "csubr6"}

// real type with different kinds
func.func @csubr7(%arg0: !fir.type<_QFcsubr7Tdt{r1:f32,r2:f64,r3:f64}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr7"} { return }
//CHECK-LABEL: func.func @csubr7(%arg0: !fir.array<3xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubr7"}

// complex type
func.func @csubc1(%arg0: !fir.type<_QFcsubc1Tdt{r1:complex<f32>,r2:complex<f32>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc1"} { return }
//CHECK-LABEL: func.func @csubc1(%arg0: !fir.array<4xf32> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc1"}

func.func @csubc2(%arg0: !fir.type<_QFcsubc2Tdt{r1:complex<f64>,r2:complex<f64>,r3:complex<f64>,r4:complex<f64>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc2"} { return }
//CHECK-LABEL: func.func @csubc2(%arg0: !fir.array<8xf64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc2"}

// complex type (> 8 floats and size < 64 bytes)
func.func @csubc3(%arg0: !fir.type<_QFcsubc3Tdt{r:!fir.array<5xcomplex<f32>>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc3"} { return }
//CHECK-LABEL: func.func @csubc3(%arg0: !fir.array<5xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc3"}

// complex type (size > 64 bytes)
func.func @csubc4(%arg0: !fir.type<_QFcsubc4Tdt{r:!fir.array<9xcomplex<f32>>}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csubc4"} { return }
//CHECK-LABEL: func.func @csubc4(%arg0: !fir.ref<!fir.type<_QFcsubc4Tdt{r:!fir.array<9xcomplex<f32>>}>> {fir.bindc_name = "arg", llvm.align = 8 : i32, llvm.byval = !fir.type<_QFcsubc4Tdt{r:!fir.array<9xcomplex<f32>>}>}) attributes {fir.bindc_name = "csubc4"}

// mixed type
func.func @csub1(%arg0: !fir.type<_QFcsub1Tdt{c:!fir.char<1>,r:f32,i:i64}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub1"} { return }
//CHECK-LABEL: func.func @csub1(%arg0: !fir.array<2xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub1"}

// nested derived types
func.func @csub2(%arg0: !fir.type<_QFcsub2Tdt1{xdt0:!fir.type<_QFcsub2Tdt0{f1:f32,f2:f32,f3:f32}>,x1:f32,x2:f32,x3:f32,x4:f32,x5:f32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub2"} { return }
//CHECK-LABEL: func.func @csub2(%arg0: !fir.array<8xf32> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub2"}

func.func @csub3(%arg0: !fir.type<_QFcsub3Tdt1{xdt0:!fir.type<_QFcsub3Tdt0{f1:f32,f2:f32,f3:f32}>,x1:f32,x2:f32,i:i32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub3"} { return }
//CHECK-LABEL: func.func @csub3(%arg0: !fir.array<3xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub3"}

func.func @csub4(%arg0: !fir.type<_QFcsub4Tdt1{xdt0:!fir.type<_QFcsub4Tdt0{f1:f32}>,x1:f64}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub4"} { return }
//CHECK-LABEL: func.func @csub4(%arg0: !fir.array<2xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub4"}

func.func @csub5(%arg0: !fir.type<_QFcsub5Tdt1{xdt0:!fir.type<_QFcsub5Tdt0{f1:complex<f32>}>,x1:f32}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub5"} { return }
//CHECK-LABEL: func.func @csub5(%arg0: !fir.array<3xf32> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub5"}

func.func @csub6(%arg0: !fir.type<_QFcsub6Tdt1{xdt0:!fir.type<_QFcsub6Tdt0{f1:complex<f32>}>,x1:f64}> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub6"} { return }
//CHECK-LABEL: func.func @csub6(%arg0: !fir.array<2xi64> {fir.bindc_name = "arg"}) attributes {fir.bindc_name = "csub6"}