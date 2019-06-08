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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

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
// CHECK-NEXT: br i1 %[[COND]], label %{{.+}}, label %[[PFORCONDCLEANUP:.+]], !llvm.loop

// CHECK: [[PFORCONDCLEANUP]]:
// CHECK-NEXT: sync within %[[SYNCREG]]
