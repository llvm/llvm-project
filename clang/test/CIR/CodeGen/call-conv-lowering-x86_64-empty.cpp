// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

struct Empty {};
struct EmptyMem { Empty e; };
struct HasEmptyBase : Empty {};
struct Derived : EmptyMem { int i; };
struct Aligned {} __attribute__((aligned(16)));
struct NoUnique { [[no_unique_address]] Empty a, b, c; };
struct NoUniqueOne { [[no_unique_address]] Empty e; };
struct UnnamedBits { int : 3; };
struct Reserved { unsigned : 32; };
struct OneByte { unsigned char c; };
struct ArrOfEmpty { Empty a[2]; };
struct HasEmpty { int x; Empty e; };
struct EmptyFirst { Empty e; int x; };
struct EmptySecond { long a; Empty e; };
struct EmptySSE { double a; Empty e; };
struct FloatEmpty { float a; Empty e; };
struct FloatEmptyFirst { Empty e; float a; };
struct alignas(32) Big32 {};
union UBits { unsigned : 3; };
union UNone {};

// An empty class is passed in no register at all.
int takeEmpty(Empty v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeEmpty5Emptyi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeEmpty5Emptyi(i32 noundef %{{[^,]+}})

// A plain empty member leaves the record non-empty under the Itanium rule, so
// this is dropped because the member contributes no eightbyte.
int takeEmptyMem(EmptyMem v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z12takeEmptyMem8EmptyMemi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z12takeEmptyMem8EmptyMemi(i32 noundef %{{[^,]+}})

// Emptiness does follow a base class.
int takeHasEmptyBase(HasEmptyBase v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z16takeHasEmptyBase12HasEmptyBasei(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z16takeHasEmptyBase12HasEmptyBasei(i32 noundef %{{[^,]+}})

// The empty base takes layout space, so the int sits at offset 4 and the
// eightbyte covering both coerces to i64.
int takeDerived(Derived v) { return v.i; }

// CIR: cir.func {{.*}}@_Z11takeDerived7Derived(%arg0: !u64i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z11takeDerived7Derived(i64 %{{[^,]+}})

// An alignment attribute widens the padding, and several [[no_unique_address]]
// members do too, but neither makes the record carry data.
int takeAligned(Aligned v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z11takeAligned7Alignedi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z11takeAligned7Alignedi(i32 noundef %{{[^,]+}})

int takeNoUnique(NoUnique v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z12takeNoUnique8NoUniquei(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z12takeNoUnique8NoUniquei(i32 noundef %{{[^,]+}})

// [[no_unique_address]] is what lets a single empty member count as empty.
int takeNoUniqueOne(NoUniqueOne v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z15takeNoUniqueOne11NoUniqueOnei(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeNoUniqueOne11NoUniqueOnei(i32 noundef %{{[^,]+}})

// Unnamed bit-field storage is marked empty rather than pad, so a record of
// nothing but unnamed bit-fields carries no data either.
int takeUnnamedBits(UnnamedBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z15takeUnnamedBits11UnnamedBitsi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeUnnamedBits11UnnamedBitsi(i32 noundef %{{[^,]+}})

int takeReserved(Reserved v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z12takeReserved8Reservedi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z12takeReserved8Reservedi(i32 noundef %{{[^,]+}})

// A byte of real data keeps its register.
int takeOneByte(OneByte v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z11takeOneByte7OneBytei(%arg0: !u8i {{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z11takeOneByte7OneBytei(i8 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// An array of empty records leaves its record non-empty, and the array still
// contributes no eightbyte.
int takeArrOfEmpty(ArrOfEmpty v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z14takeArrOfEmpty10ArrOfEmptyi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z14takeArrOfEmpty10ArrOfEmptyi(i32 noundef %{{[^,]+}})

// An empty member contributes no eightbyte, so only the int is classified.
int takeHasEmpty(HasEmpty v) { return v.x; }

// CIR: cir.func {{.*}}@_Z12takeHasEmpty8HasEmpty(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z12takeHasEmpty8HasEmpty(i32 %{{[^,]+}})

int takeEmptyFirst(EmptyFirst v) { return v.x; }

// CIR: cir.func {{.*}}@_Z14takeEmptyFirst10EmptyFirst(%arg0: !u64i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z14takeEmptyFirst10EmptyFirst(i64 %{{[^,]+}})

// The empty member owns the second eightbyte alone, which classifies NoClass
// and is dropped rather than merged into a register of its own.
int takeEmptySecond(EmptySecond v) { return 0; }

// CIR: cir.func {{.*}}@_Z15takeEmptySecond11EmptySecond(%arg0: !s64i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeEmptySecond11EmptySecond(i64 %{{[^,]+}})

// The same with an SSE eightbyte below it.
double takeEmptySSE(EmptySSE v) { return v.a; }

// CIR: cir.func {{.*}}@_Z12takeEmptySSE8EmptySSE(%arg0: !cir.double {{.*}}) -> (!cir.double
// LLVM: define dso_local noundef double @_Z12takeEmptySSE8EmptySSE(double %{{[^,]+}})

// Here the empty member shares eightbyte 0 with the float rather than owning
// one, so NoClass has to survive a merge against SSE instead of standing alone.
float takeFloatEmpty(FloatEmpty v) { return v.a; }

// CIR: cir.func {{.*}}@_Z14takeFloatEmpty10FloatEmpty(%arg0: !cir.float {{.*}}) -> (!cir.float
// LLVM: define dso_local noundef float @_Z14takeFloatEmpty10FloatEmpty(float %{{[^,]+}})

// The empty member first pushes the float to offset 4, and the eightbyte
// covering both coerces to double.
float takeFloatEmptyFirst(FloatEmptyFirst v) { return v.a; }

// CIR: cir.func {{.*}}@_Z19takeFloatEmptyFirst15FloatEmptyFirst(%arg0: !cir.double {{.*}}) -> (!cir.float
// LLVM: define dso_local noundef float @_Z19takeFloatEmptyFirst15FloatEmptyFirst(double %{{[^,]+}})

// Past two eightbytes SysV says memory whatever the content, so an empty class
// this size is passed indirectly at its declared alignment.
int takeBig32(Big32 v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeBig325Big32i(%arg0: !cir.ptr<!rec_Big32> {llvm.align = 32 : i64, llvm.byval = !rec_Big32, llvm.noalias, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z9takeBig325Big32i(ptr noalias noundef byval(%struct.Big32) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z9takeBig325Big32i(ptr noundef byval(%struct.Big32) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// The same class returned uses sret at that alignment.
Big32 retBig32() { return Big32{}; }

// CIR: cir.func {{.*}}@_Z8retBig32v(%arg0: !cir.ptr<!rec_Big32> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Big32, llvm.writable}
// LLVM: define dso_local void @_Z8retBig32v(ptr dead_on_unwind noalias writable sret(%struct.Big32) align 32 %{{[^,]+}})

// A union of only unnamed bit-fields holds no data, and neither does one with
// no members at all.
int takeUBits(UBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeUBits5UBitsi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeUBits5UBitsi(i32 noundef %{{[^,]+}})

int takeUNone(UNone v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeUNone5UNonei(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeUNone5UNonei(i32 noundef %{{[^,]+}})

// An empty return is dropped to void.
Empty retEmpty() { return Empty{}; }

// CIR: cir.func {{.*}}@_Z8retEmptyv()
// LLVM: define dso_local void @_Z8retEmptyv()

// A call site drops the operand as well as the parameter.
int caller(int k) {
  Empty e;
  return takeEmpty(e, k);
}

// CIR: cir.func {{.*}}@_Z6calleri(%arg0: !s32i {{.*}}) -> (!s32i
// CIR:   cir.call @_Z9takeEmpty5Emptyi(%{{[0-9]+}}) : (!s32i {llvm.noundef}) -> (!s32i {llvm.noundef})
// LLVM: define dso_local noundef i32 @_Z6calleri(i32 noundef %{{[^,]+}})
// LLVM:   call noundef i32 @_Z9takeEmpty5Emptyi(i32 noundef %{{[^,]+}})
