// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Empty {};
struct EmptyMem { Empty e; };
struct HasEmptyBase : Empty {};
struct Derived : EmptyMem { int i; };
struct Aligned {} __attribute__((aligned(16)));
struct NoUnique { [[no_unique_address]] Empty a, b, c; };
struct NoUniqueOne { [[no_unique_address]] Empty e; };
struct UnnamedBits { int : 3; };
struct Reserved { unsigned : 32; };
struct ReservedBase : Reserved { int i; };
struct ReservedMem { [[no_unique_address]] Reserved r; int i; };
struct OneByte { unsigned char c; };
struct ArrOfEmpty { Empty a[2]; };
struct HasEmpty { int x; Empty e; };
struct EmptyFirst { Empty e; int x; };
struct EmptySecond { long a; Empty e; };
struct EmptySSE { double a; Empty e; };
struct FloatEmpty { float a; Empty e; };
struct FloatEmptyFirst { Empty e; float a; };
struct alignas(32) Big32 {};
struct EBits { int : 0; };
struct HoldsEmptyBits { EBits e; int i; };
union UBits { unsigned : 3; };
union UNone {};
union UEmptyInt { Empty e; int i; };
union UEmptyAligned { Aligned e; int i; };
union UArrEmpty { Empty a[2]; char c; };
union UEmptyOnly { Empty e; };
union UEmptyDouble { Empty e; double d; };
union UEmptyBytes { Empty e; char c[8]; };
union UBigEmpty { Big32 e; int i; };
union UEmptyBaseMem { HasEmptyBase e; int i; };
union UValue { Empty mono; int i; long long ll; double d; const char *s; };
struct ArgStore { UValue value; unsigned char type; };

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

// Unnamed bit-field storage is marked empty because no field of the source
// reads it, but the classifier gives it the eightbyte classes of a named
// bit-field's, so a record of nothing but unnamed bit-fields is still passed.
int takeUnnamedBits(UnnamedBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z15takeUnnamedBits11UnnamedBitsi(%arg0: !u8i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeUnnamedBits11UnnamedBitsi(i8 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// The eightbyte is coerced from the access unit, so a wider reservation is
// passed in a wider register.
int takeReserved(Reserved v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z12takeReserved8Reservedi(%arg0: !u32i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z12takeReserved8Reservedi(i32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// A record reaches that storage through a base the same way it reaches a data
// member, so the eightbyte covering both is an integer one.
int takeReservedBase(ReservedBase v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z16takeReservedBase12ReservedBasei(%arg0: !u64i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z16takeReservedBase12ReservedBasei(i64 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// And through a [[no_unique_address]] member, which does not make a record
// holding an unnamed bit-field empty.
int takeReservedMem(ReservedMem v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z15takeReservedMem11ReservedMemi(%arg0: !u64i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeReservedMem11ReservedMemi(i64 %{{[^,]+}}, i32 noundef %{{[^,]+}})

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

// CIR: cir.func {{.*}}@_Z9takeBig325Big32i(%arg0: !cir.ptr<!rec_Big32> {llvm.align = 32 : i64, llvm.byval = !rec_Big32, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeBig325Big32i(ptr noundef byval(%struct.Big32) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// The same class returned uses sret at that alignment.
Big32 retBig32() { return Big32{}; }

// CIR: cir.func {{.*}}@_Z8retBig32v(%arg0: !cir.ptr<!rec_Big32> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Big32, llvm.writable}
// LLVM: define dso_local void @_Z8retBig32v(ptr dead_on_unwind noalias writable sret(%struct.Big32) align 32 %{{[^,]+}})

// A zero-width unnamed bit-field reserves no storage for the classifier to
// coerce from, so unlike a wider reservation this record is dropped.
int takeEmptyEBits(EBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z14takeEmptyEBits5EBitsi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z14takeEmptyEBits5EBitsi(i32 noundef %{{[^,]+}})

// It still takes layout space as a member, so the int sits at offset 4 and the
// eightbyte covering both coerces to i64.
int takeHoldsEmptyBits(HoldsEmptyBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z18takeHoldsEmptyBits14HoldsEmptyBitsi(%arg0: !u64i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z18takeHoldsEmptyBits14HoldsEmptyBitsi(i64 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// A union variant that is an unnamed bit-field holds data the same way a
// struct member does, so only a union with no members at all holds none.
int takeUBits(UBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeUBits5UBitsi(%arg0: !u8i{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeUBits5UBitsi(i8 %{{[^,]+}}, i32 noundef %{{[^,]+}})

int takeUNone(UNone v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z9takeUNone5UNonei(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z9takeUNone5UNonei(i32 noundef %{{[^,]+}})

// A union with an empty member coerces from the member that supplies bytes,
// not from the union's size.
int takeUEmptyInt(UEmptyInt v) { return v.i; }

// CIR: cir.func {{.*}}@_Z13takeUEmptyInt9UEmptyInt(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z13takeUEmptyInt9UEmptyInt(i32 %{{[^,]+}})

// Here the empty member's alignment (16) outranks the int's (4), so the same
// rule matters more: a member supplying no bytes still cannot decide the
// storage type, and the 16-byte union coerces to the int's eightbyte rather
// than widening to i64.
int takeUEmptyAligned(UEmptyAligned v) { return v.i; }

// CIR: cir.func {{.*}}@_Z17takeUEmptyAligned13UEmptyAligned(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z17takeUEmptyAligned13UEmptyAligned(i32 %{{[^,]+}})

// An alignment tie is broken by size, so an array of empty records outranks
// the byte of data while holding none itself.
int takeUArrEmpty(UArrEmpty v) { return v.c; }

// CIR: cir.func {{.*}}@_Z13takeUArrEmpty9UArrEmpty(%arg0: !s8i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z13takeUArrEmpty9UArrEmpty(i8 %{{[^,]+}})

// A union of nothing but an empty member is dropped, like an empty class.
int takeUEmptyOnly(UEmptyOnly v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z14takeUEmptyOnly10UEmptyOnlyi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z14takeUEmptyOnly10UEmptyOnlyi(i32 noundef %{{[^,]+}})

// Skipping the empty member leaves the class to the member that remains, so
// this passes in an SSE register rather than an integer one.
double takeUEmptyDouble(UEmptyDouble v) { return v.d; }

// CIR: cir.func {{.*}}@_Z16takeUEmptyDouble12UEmptyDouble(%arg0: !cir.double {{.*}}) -> (!cir.double
// LLVM: define dso_local noundef double @_Z16takeUEmptyDouble12UEmptyDouble(double %{{[^,]+}})

// Where the data member fills the eightbyte there is nothing to narrow.
int takeUEmptyBytes(UEmptyBytes v) { return v.c[0]; }

// CIR: cir.func {{.*}}@_Z15takeUEmptyBytes11UEmptyBytes(%arg0: !u64i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeUEmptyBytes11UEmptyBytes(i64 %{{[^,]+}})

// Past two eightbytes SysV says memory whatever the content, so the empty
// member changes nothing here.
int takeUBigEmpty(UBigEmpty v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z13takeUBigEmpty9UBigEmptyi(%arg0: !cir.ptr<!rec_UBigEmpty> {llvm.align = 32 : i64, llvm.byval = !rec_UBigEmpty, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z13takeUBigEmpty9UBigEmptyi(ptr noundef byval(%union.UBigEmpty) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// The same union returned uses sret at that alignment.
UBigEmpty retUBigEmpty() { return UBigEmpty{}; }

// CIR: cir.func {{.*}}@_Z12retUBigEmptyv(%arg0: !cir.ptr<!rec_UBigEmpty> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_UBigEmpty, llvm.writable}
// LLVM: define dso_local void @_Z12retUBigEmptyv(ptr dead_on_unwind noalias writable sret(%union.UBigEmpty) align 32 %{{[^,]+}})

// Emptiness reaches the union member through a base class as well.
int takeUEmptyBaseMem(UEmptyBaseMem v) { return v.i; }

// CIR: cir.func {{.*}}@_Z17takeUEmptyBaseMem13UEmptyBaseMem(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z17takeUEmptyBaseMem13UEmptyBaseMem(i32 %{{[^,]+}})

// Several scalars alongside one empty member: the widest of the scalars decides
// the coercion.
long long takeUValue(UValue v) { return v.ll; }

// CIR: cir.func {{.*}}@_Z10takeUValue6UValue(%arg0: !s64i {{.*}}) -> (!s64i
// LLVM: define dso_local noundef i64 @_Z10takeUValue6UValue(i64 %{{[^,]+}})

// The same union as a struct member, where the struct's eightbytes are what
// gets classified.
long long takeArgStore(ArgStore a) { return a.value.ll; }

// CIR: cir.func {{.*}}@_Z12takeArgStore8ArgStore(%arg0: !s64i {{.*}}, %arg1: !u8i {{.*}}) -> (!s64i
// LLVM: define dso_local noundef i64 @_Z12takeArgStore8ArgStore(i64 %{{[^,]+}}, i8 %{{[^,]+}})

// The union returned by value round-trips through its coercion.
UValue retUValue() { return UValue{}; }

// CIR: cir.func {{.*}}@_Z9retUValuev() -> !s64i
// LLVM: define dso_local i64 @_Z9retUValuev()

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
