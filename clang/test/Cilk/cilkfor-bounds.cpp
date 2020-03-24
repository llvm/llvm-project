// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-linux-gnu -fcilkplus -ftapir=none -verify -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

typedef __SIZE_TYPE__ size_t;

void bar(size_t i);

void up(size_t start, size_t end) {
  _Cilk_for (size_t i = start; i < end; ++i)
    bar(i);
}

// CHECK-LABEL: define void @_Z2upmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ult i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], 1
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ult i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_leq(size_t start, size_t end) {
  _Cilk_for (size_t i = start; i <= end; ++i)
    bar(i);
}

// CHECK-LABEL: define void @_Z6up_leqmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ule i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], 1
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ule i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD2:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_flip(size_t start, size_t end) {
  _Cilk_for (size_t i = start; end > i; ++i)
    bar(i);
}

// CHECK-LABEL: define void @_Z7up_flipmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ugt i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], 1
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ugt i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD3:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_flip_geq(size_t start, size_t end) {
  _Cilk_for (size_t i = start; end >= i; ++i)
    bar(i);
}

// CHECK-LABEL: define void @_Z11up_flip_geqmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp uge i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], 1
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp uge i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD4:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_stride(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; i < end; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z9up_stridemmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ult i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], %[[STRIDE]]
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ult i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD5:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_stride_leq(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; i <= end; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z13up_stride_leqmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ule i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ule i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD6:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_stride_flip(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; end > i; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z14up_stride_flipmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ugt i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], %[[STRIDE]]
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ugt i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD7:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_stride_flip_geq(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; end >= i; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z18up_stride_flip_geqmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp uge i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp uge i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD8:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_ne_stride(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; i != end; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z12up_ne_stridemmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ne i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ne i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD8:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void up_ne_stride_flip(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = start; end != i; i += stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z17up_ne_stride_flipmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ne i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDLIMIT]], %[[ENDINIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = add i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ne i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD9:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down(size_t start, size_t end) {
  _Cilk_for (size_t i = end; i > start; --i)
    bar(i);
}

// CHECK-LABEL: define void @_Z4downmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ugt i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], 1
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ugt i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD10:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_geq(size_t start, size_t end) {
  _Cilk_for (size_t i = end; i >= start; --i)
    bar(i);
}

// CHECK-LABEL: define void @_Z8down_geqmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp uge i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], 1
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp uge i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD11:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_flip(size_t start, size_t end) {
  _Cilk_for (size_t i = end; start < i; --i)
    bar(i);
}

// CHECK-LABEL: define void @_Z9down_flipmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ult i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], 1
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ult i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD12:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_flip_leq(size_t start, size_t end) {
  _Cilk_for (size_t i = end; start <= i; --i)
    bar(i);
}

// CHECK-LABEL: define void @_Z13down_flip_leqmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ule i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], 1
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], 1
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ule i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD13:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_stride(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; i > start; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z11down_stridemmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ugt i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], %[[STRIDE]]
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ugt i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD14:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_stride_geq(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; i >= start; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z15down_stride_geqmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp uge i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp uge i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD15:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_stride_flip(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; start < i; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z16down_stride_flipmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ult i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[ENDSUB1:.+]] = sub i64 %[[ENDSUB]], 1
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB1]], %[[STRIDE]]
// CHECK-NEXT: %[[ENDADD:.+]] = add i64 %[[ENDDIV]], 1
// CHECK-NEXT: store i64 %[[ENDADD]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ult i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD16:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_stride_flip_leq(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; start <= i; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z20down_stride_flip_leqmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ule i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ule i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD17:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

void down_ne_stride(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; i != start; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z14down_ne_stridemmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ne i64 %[[INITCMPINIT]], %[[INITCMPLIMIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[COND:.+]] = icmp ne i64 %[[CONDEND]], %[[CONDBEGIN]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD18:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]
				   
void down_ne_stride_flip(size_t start, size_t end, size_t stride) {
  _Cilk_for (size_t i = end; start != i; i -= stride)
    bar(i);
}

// CHECK-LABEL: define void @_Z19down_ne_stride_flipmmm(

// CHECK: %[[START:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[START]], i64* %[[INIT:.+]], align 8
// CHECK-NEXT: %[[END:.+]] = load i64, i64*
// CHECK-NEXT: store i64 %[[END]], i64* %[[LIMIT:.+]], align 8
// CHECK-NEXT: %[[INITCMPLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[INITCMPINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[INITCMP:.+]] = icmp ne i64 %[[INITCMPLIMIT]], %[[INITCMPINIT]]
// CHECK-NEXT: br i1 %[[INITCMP]], label %[[PFORPH:.+]], label %[[PFOREND:.+]]

// CHECK: [[PFORPH]]:
// CHECK-NEXT: store i64 0, i64* %[[BEGIN:.+]], align 8
// CHECK-NEXT: %[[ENDINIT:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[ENDLIMIT:.+]] = load i64, i64* %[[LIMIT]]
// CHECK-NEXT: %[[ENDSUB:.+]] = sub i64 %[[ENDINIT]], %[[ENDLIMIT]]
// CHECK-NEXT: %[[STRIDE:.+]] = load i64, i64* %[[STRIDEADDR:.+]], align 8
// CHECK-NEXT: %[[ENDDIV:.+]] = udiv i64 %[[ENDSUB]], %[[STRIDE]]
// CHECK-NEXT: store i64 %[[ENDDIV]], i64* %[[END:.+]], align 8

// CHECK: %[[INITITER:.+]] = load i64, i64* %[[INIT]]
// CHECK-NEXT: %[[BEGINITER:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[STRIDEITER:.+]] = load i64, i64* %[[STRIDEADDR]]
// CHECK-NEXT: %[[ITERMUL:.+]] = mul i64 %[[BEGINITER]], %[[STRIDEITER]]
// CHECK-NEXT: %[[ITERADD:.+]] = sub i64 %[[INITITER]], %[[ITERMUL]]
// CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[PFORINC:.+]]

// CHECK: [[DETACHED]]:
// CHECK: %[[ITERSLOT:.+]] = alloca i64, align 8
// CHECK: store i64 %[[ITERADD]], i64* %[[ITERSLOT]]

// CHECK: [[PFORINC]]:
// CHECK-NEXT: %[[INCBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[INC:.+]] = add i64 %[[INCBEGIN]], 1
// CHECK-NEXT: store i64 %[[INC]], i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDBEGIN:.+]] = load i64, i64* %[[BEGIN]]
// CHECK-NEXT: %[[CONDEND:.+]] = load i64, i64* %[[END]]
// CHECK-NEXT: %[[COND:.+]] = icmp ne i64 %[[CONDBEGIN]], %[[CONDEND]]
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop ![[LOOPMD19:.+]]

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

// CHECK: ![[LOOPMD]] = distinct !{![[LOOPMD]], ![[SPAWNSTRATEGY:.+]]}
// CHECK: ![[SPAWNSTRATEGY]] = !{!"tapir.loop.spawn.strategy", i32 1}
// CHECK: ![[LOOPMD2]] = distinct !{![[LOOPMD2]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD3]] = distinct !{![[LOOPMD3]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD4]] = distinct !{![[LOOPMD4]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD5]] = distinct !{![[LOOPMD5]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD6]] = distinct !{![[LOOPMD6]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD7]] = distinct !{![[LOOPMD7]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD8]] = distinct !{![[LOOPMD8]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD9]] = distinct !{![[LOOPMD9]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD10]] = distinct !{![[LOOPMD10]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD11]] = distinct !{![[LOOPMD11]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD12]] = distinct !{![[LOOPMD12]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD13]] = distinct !{![[LOOPMD13]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD14]] = distinct !{![[LOOPMD14]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD15]] = distinct !{![[LOOPMD15]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD16]] = distinct !{![[LOOPMD16]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD17]] = distinct !{![[LOOPMD17]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD18]] = distinct !{![[LOOPMD18]], ![[SPAWNSTRATEGY]]}
// CHECK: ![[LOOPMD19]] = distinct !{![[LOOPMD19]], ![[SPAWNSTRATEGY]]}
