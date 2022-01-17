// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -o %t-1.ll %s
// RUN: FileCheck --check-prefix=ALL -check-prefix SANE --input-file=%t-1.ll %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -fno-assume-sane-operator-new -o %t-2.ll %s
// RUN: FileCheck --check-prefix=ALL -check-prefix SANENOT --input-file=%t-2.ll %s

class teste {
  int A;
public:
  teste() : A(2) {}
};

void f1() {
  // ALL: declare noundef nonnull i8* @_Znwj(
  new teste();
}

// rdar://5739832 - operator new should check for overflow in multiply.
void *f2(long N) {
  return new int[N];

  // ALL:      [[UWO:%.*]] = call {{.*}} @llvm.umul.with.overflow
  // ALL-NEXT: [[OVER:%.*]] = extractvalue {{.*}} [[UWO]], 1
  // ALL-NEXT: [[SUM:%.*]] = extractvalue {{.*}} [[UWO]], 0
  // ALL-NEXT: [[RESULT:%.*]] = select i1 [[OVER]], i32 -1, i32 [[SUM]]
  // SANE-NEXT: call noalias noundef nonnull i8* @_Znaj(i32 noundef [[RESULT]])
  // SANENOT-NEXT: call noundef nonnull i8* @_Znaj(i32 noundef [[RESULT]])
}

// ALL: declare noundef nonnull i8* @_Znaj(
