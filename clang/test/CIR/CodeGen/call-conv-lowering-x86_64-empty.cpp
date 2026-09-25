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
struct HiWord { Empty e; long hi; };
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
union UNuaEmptyInt { [[no_unique_address]] Empty e; int i; };
union UNuaEmptyAligned { [[no_unique_address]] Aligned e; int i; };
union UNuaEmptyOnly { [[no_unique_address]] Empty e; };
union UNuaEmptyDouble { [[no_unique_address]] Empty e; double d; };
union UNuaBigEmpty { [[no_unique_address]] Big32 e; int i; };
union UNuaEmptyBaseMem { [[no_unique_address]] HasEmptyBase e; int i; };
union UNuaEmptyUnnamedBits { [[no_unique_address]] Empty e; int : 24; };
union UNuaEmptyBitInt { [[no_unique_address]] Empty e; int b : 20; int i; };
union UNuaEmptyAlignedBits { [[no_unique_address]] Aligned e; int b : 8; };
union UNuaEmptyNarrow { [[no_unique_address]] Empty e; short s; };
struct SFloatPair { float a, b; };
union UNuaEmptyFloats { [[no_unique_address]] Empty e; SFloatPair f; };
union UNuaEmptyBytes16 { [[no_unique_address]] Empty e; char c[16]; };
union UNuaEmptyNarrowHi { [[no_unique_address]] Aligned e; char c[9]; };
union UNuaBigEmptyOnly { [[no_unique_address]] Big32 e; };
union UZeroLenArr { int x[0]; int i; };
union UZeroLenOnly { int x[0]; };
union UNuaNoRegs { [[no_unique_address]] Empty e; int i; ~UNuaNoRegs(); };
struct NuaHiWord { UNuaEmptyOnly u; long hi; };
struct NuaExpected {
  [[no_unique_address]] UNuaEmptyInt u;
  [[no_unique_address]] bool has;
};

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

// Here the empty member owns eightbyte 0 rather than eightbyte 1, so NoClass
// cannot just be dropped: the coercion has to start at byte 8.
long takeHiWord(HiWord v) { return v.hi; }

// CIR: cir.func {{.*}}@_Z10takeHiWord6HiWord(%arg0: !s64i {{.*}}) -> (!s64i
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce"
// CIR:   %[[U8:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_HiWord> -> !cir.ptr<!u8i>
// CIR:   %[[OFF:.+]] = cir.const #cir.int<8> : !s64i
// CIR:   %[[GEP:.+]] = cir.ptr_stride %[[U8]], %[[OFF]]
// CIR:   %[[HI:.+]] = cir.cast bitcast %[[GEP]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   cir.store %arg0, %[[HI]] : !s64i, !cir.ptr<!s64i>
// LLVM: define dso_local noundef i64 @_Z10takeHiWord6HiWord(i64 %[[ARG:[^)]+]])
// LLVM:   %[[SLOT:.+]] = alloca %struct.HiWord, align 8
// LLVM:   %[[HI:.+]] = getelementptr{{( inbounds)?}} i8, ptr %[[SLOT]], i64 8
// LLVM:   store i64 %[[ARG]], ptr %[[HI]], align 8

// The same offset on the return side.
HiWord giveHiWord(long hi) {
  HiWord w;
  w.hi = hi;
  return w;
}

// CIR: cir.func {{.*}}@_Z10giveHiWordl(%arg0: !s64i {llvm.noundef} {{.*}}) -> !s64i
// LLVM: define dso_local i64 @_Z10giveHiWordl(i64 noundef %{{[^,]+}})
// LLVM:   %[[RGEP:.+]] = getelementptr{{( inbounds)?}} i8, ptr %{{.+}}, i64 8
// LLVM:   %[[RVAL:.+]] = load i64, ptr %[[RGEP]], align 8
// LLVM:   ret i64 %[[RVAL]]

// Caller-side coercion, on the return received and the argument passed.
long callerHiWord(long hi) {
  return takeHiWord(giveHiWord(hi));
}

// CIR: cir.func {{.*}}@_Z12callerHiWordl(%arg0: !s64i {{.*}}) -> (!s64i
// CIR:   %[[RET:.+]] = cir.call @_Z10giveHiWordl(
// CIR:   %[[ROFF:.+]] = cir.const #cir.int<8> : !s64i
// CIR:   %[[RGEP:.+]] = cir.ptr_stride %{{.+}}, %[[ROFF]]
// CIR:   %[[RPTR:.+]] = cir.cast bitcast %[[RGEP]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   cir.store %[[RET]], %[[RPTR]] : !s64i, !cir.ptr<!s64i>
// CIR:   %[[AOFF:.+]] = cir.const #cir.int<8> : !s64i
// CIR:   %[[AGEP:.+]] = cir.ptr_stride %{{.+}}, %[[AOFF]]
// CIR:   %[[APTR:.+]] = cir.cast bitcast %[[AGEP]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   %[[AVAL:.+]] = cir.load %[[APTR]] : !cir.ptr<!s64i>, !s64i
// CIR:   %{{.+}} = cir.call @_Z10takeHiWord6HiWord(%[[AVAL]])
// LLVM: define dso_local noundef i64 @_Z12callerHiWordl(i64 noundef %{{[^,)]+}})
// LLVM:   %[[RET:.+]] = call i64 @_Z10giveHiWordl(i64 noundef %{{.+}})
// LLVM:   %[[RSLOT:.+]] = getelementptr{{( inbounds)?}} i8, ptr %{{.+}}, i64 8
// LLVM:   store i64 %[[RET]], ptr %[[RSLOT]], align 8
// LLVM:   %[[ASLOT:.+]] = getelementptr{{( inbounds)?}} i8, ptr %{{.+}}, i64 8
// LLVM:   %[[AVAL:.+]] = load i64, ptr %[[ASLOT]], align 8
// LLVM:   %{{.+}} = call noundef i64 @_Z10takeHiWord6HiWord(i64 %[[AVAL]])

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

// A variant of empty class type is marked data unless [[no_unique_address]]
// makes CIRGen mark it empty.  Neither mark reaches the classifier, so where
// a case below repeats a shape from above it classifies the same either way.
int takeUNuaEmptyInt(UNuaEmptyInt v) { return v.i; }

// CIR: cir.func {{.*}}@_Z16takeUNuaEmptyInt12UNuaEmptyInt(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z16takeUNuaEmptyInt12UNuaEmptyInt(i32 %{{[^,]+}})

int takeUNuaEmptyAligned(UNuaEmptyAligned v) { return v.i; }

// CIR: cir.func {{.*}}@_Z20takeUNuaEmptyAligned16UNuaEmptyAligned(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z20takeUNuaEmptyAligned16UNuaEmptyAligned(i32 %{{[^,]+}})

int takeUNuaEmptyOnly(UNuaEmptyOnly v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z17takeUNuaEmptyOnly13UNuaEmptyOnlyi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z17takeUNuaEmptyOnly13UNuaEmptyOnlyi(i32 noundef %{{[^,]+}})

double takeUNuaEmptyDouble(UNuaEmptyDouble v) { return v.d; }

// CIR: cir.func {{.*}}@_Z19takeUNuaEmptyDouble15UNuaEmptyDouble(%arg0: !cir.double {{.*}}) -> (!cir.double
// LLVM: define dso_local noundef double @_Z19takeUNuaEmptyDouble15UNuaEmptyDouble(double %{{[^,]+}})

int takeUNuaBigEmpty(UNuaBigEmpty v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z16takeUNuaBigEmpty12UNuaBigEmptyi(%arg0: !cir.ptr<!rec_UNuaBigEmpty> {llvm.align = 32 : i64, llvm.byval = !rec_UNuaBigEmpty, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z16takeUNuaBigEmpty12UNuaBigEmptyi(ptr noundef byval(%union.UNuaBigEmpty) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

UNuaBigEmpty retUNuaBigEmpty() { return UNuaBigEmpty{}; }

// CIR: cir.func {{.*}}@_Z15retUNuaBigEmptyv(%arg0: !cir.ptr<!rec_UNuaBigEmpty> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_UNuaBigEmpty, llvm.writable}
// LLVM: define dso_local void @_Z15retUNuaBigEmptyv(ptr dead_on_unwind noalias writable sret(%union.UNuaBigEmpty) align 32 %{{[^,]+}})

// A union of nothing but an over-aligned empty variant is still sized by it,
// so past two eightbytes it goes to memory rather than being dropped.
int takeUNuaBigEmptyOnly(UNuaBigEmptyOnly v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z20takeUNuaBigEmptyOnly16UNuaBigEmptyOnlyi(%arg0: !cir.ptr<!rec_UNuaBigEmptyOnly> {llvm.align = 32 : i64, llvm.byval = !rec_UNuaBigEmptyOnly, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z20takeUNuaBigEmptyOnly16UNuaBigEmptyOnlyi(ptr noundef byval(%union.UNuaBigEmptyOnly) align 32 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// A zero-length array variant is marked empty and occupies nothing, so
// dropping it cannot lose data and the int decides.
int takeUZeroLenArr(UZeroLenArr v) { return v.i; }

// CIR: cir.func {{.*}}@_Z15takeUZeroLenArr11UZeroLenArr(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z15takeUZeroLenArr11UZeroLenArr(i32 %{{[^,]+}})

int takeUZeroLenOnly(UZeroLenOnly v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z16takeUZeroLenOnly12UZeroLenOnlyi(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z16takeUZeroLenOnly12UZeroLenOnlyi(i32 noundef %{{[^,]+}})

// A destructor forces the union indirect without byval, so the argument is a
// pointer to the caller's own storage rather than a coercion.
int takeUNuaNoRegs(UNuaNoRegs v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z14takeUNuaNoRegs10UNuaNoRegsi(%arg0: !cir.ptr<!rec_UNuaNoRegs> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z14takeUNuaNoRegs10UNuaNoRegsi(ptr nofreeobj noundef align 4 dereferenceable(4) %{{[^,]+}}, i32 noundef %{{[^,]+}})

// The caller copies into a temporary and forwards that, since the callee
// works on the object it is handed.
int callTakeUNuaNoRegs(int k) {
  UNuaNoRegs v;
  return takeUNuaNoRegs(v, k);
}

// CIR: cir.func {{.*}}@_Z18callTakeUNuaNoRegsi
// CIR:   cir.call @_Z14takeUNuaNoRegs10UNuaNoRegsi(%{{.+}}, %{{.+}}) : (!cir.ptr<!rec_UNuaNoRegs> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}, !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z18callTakeUNuaNoRegsi(i32 noundef %{{[^,]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP:[^,]+]], ptr align 4 %{{[^,]+}}, i64 4, i1 false)
// LLVM:   call noundef i32 @_Z14takeUNuaNoRegs10UNuaNoRegsi(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP]], i32 noundef %{{[^,]+}})

// The same union returned uses sret.
UNuaNoRegs retUNuaNoRegs();
UNuaNoRegs callRetUNuaNoRegs() { return retUNuaNoRegs(); }

// CIR: cir.func {{.*}}@_Z17callRetUNuaNoRegsv(%arg0: !cir.ptr<!rec_UNuaNoRegs> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_UNuaNoRegs, llvm.writable}
// LLVM: define dso_local void @_Z17callRetUNuaNoRegsv(ptr dead_on_unwind noalias writable sret(%union.UNuaNoRegs) align 4 %{{[^,]+}})
// LLVM:   call void @_Z13retUNuaNoRegsv(ptr dead_on_unwind writable sret(%union.UNuaNoRegs) align 4 %{{[^,)]+}})

int takeUNuaEmptyBaseMem(UNuaEmptyBaseMem v) { return v.i; }

// CIR: cir.func {{.*}}@_Z20takeUNuaEmptyBaseMem16UNuaEmptyBaseMem(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z20takeUNuaEmptyBaseMem16UNuaEmptyBaseMem(i32 %{{[^,]+}})

// An unnamed bit-field variant holds data for the ABI, so it is the storage the
// coercion reads while the empty variant contributes nothing.
int takeUNuaEmptyUnnamedBits(UNuaEmptyUnnamedBits v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z24takeUNuaEmptyUnnamedBits20UNuaEmptyUnnamedBitsi(%arg0: !cir.int<u, 24>{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z24takeUNuaEmptyUnnamedBits20UNuaEmptyUnnamedBitsi(i24 %{{[^,]+}}, i32 noundef %{{[^,]+}})

// A named access unit narrower than the int it holds, alongside the empty
// variant.
int takeUNuaEmptyBitInt(UNuaEmptyBitInt v) { return v.i; }

// CIR: cir.func {{.*}}@_Z19takeUNuaEmptyBitInt15UNuaEmptyBitInt(%arg0: !s32i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z19takeUNuaEmptyBitInt15UNuaEmptyBitInt(i32 %{{[^,]+}})

// The empty variant's alignment alone sets the union's 16 bytes, so the one
// byte the access unit holds decides the eightbyte and the rest is padding.
int takeUNuaEmptyAlignedBits(UNuaEmptyAlignedBits v) { return v.b; }

// CIR: cir.func {{.*}}@_Z24takeUNuaEmptyAlignedBits20UNuaEmptyAlignedBits(%arg0: !u64i {{.*}}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z24takeUNuaEmptyAlignedBits20UNuaEmptyAlignedBits(i64 %{{[^,]+}})

// The empty variant supplies no bytes, so the short sizes the eightbyte even
// though the union's declared size rounds past it.
short takeUNuaEmptyNarrow(UNuaEmptyNarrow v) { return v.s; }

// CIR: cir.func {{.*}}@_Z19takeUNuaEmptyNarrow15UNuaEmptyNarrow(%arg0: !s16i {{.*}}) -> (!s16i
// LLVM: define dso_local noundef signext i16 @_Z19takeUNuaEmptyNarrow15UNuaEmptyNarrow(i16 %{{[^,]+}})

// A pair of floats in one eightbyte still coerces to a vector, not an integer.
float takeUNuaEmptyFloats(UNuaEmptyFloats v) { return v.f.a; }

// CIR: cir.func {{.*}}@_Z19takeUNuaEmptyFloats15UNuaEmptyFloats(%arg0: !cir.vector<2 x !cir.float> {{.*}}) -> (!cir.float
// LLVM: define dso_local noundef float @_Z19takeUNuaEmptyFloats15UNuaEmptyFloats(<2 x float> %{{[^,]+}})

// The empty variant does not disturb a multi-eightbyte coercion.
char takeUNuaEmptyBytes16(UNuaEmptyBytes16 v) { return v.c[0]; }

// CIR: cir.func {{.*}}@_Z20takeUNuaEmptyBytes1616UNuaEmptyBytes16(%arg0: !u64i {{.*}}, %arg1: !u64i {{.*}}) -> (!s8i
// LLVM: define dso_local noundef signext i8 @_Z20takeUNuaEmptyBytes1616UNuaEmptyBytes16(i64 %{{[^,]+}}, i64 %{{[^,]+}})

// Here the empty variant is what reaches the second eightbyte, which the
// data variant covers only one byte of, so the high half narrows to i8.
char takeUNuaEmptyNarrowHi(UNuaEmptyNarrowHi v) { return v.c[0]; }

// CIR: cir.func {{.*}}@_Z21takeUNuaEmptyNarrowHi17UNuaEmptyNarrowHi(%arg0: !u64i {{.*}}, %arg1: !s8i {{.*}}) -> (!s8i
// LLVM: define dso_local noundef signext i8 @_Z21takeUNuaEmptyNarrowHi17UNuaEmptyNarrowHi(i64 %{{[^,]+}}, i8 %{{[^,]+}})

// An all-empty union owns eightbyte 0, so as with takeHiWord the coercion has
// to start at byte 8 rather than dropping the NoClass half.
long takeNuaHiWord(NuaHiWord v) { return v.hi; }

// CIR: cir.func {{.*}}@_Z13takeNuaHiWord9NuaHiWord(%arg0: !s64i {{.*}}) -> (!s64i
// CIR:   %[[NSLOT:.+]] = cir.alloca "coerce"
// CIR:   %[[NU8:.+]] = cir.cast bitcast %[[NSLOT]] : !cir.ptr<!rec_NuaHiWord> -> !cir.ptr<!u8i>
// CIR:   %[[NOFF:.+]] = cir.const #cir.int<8> : !s64i
// CIR:   %[[NGEP:.+]] = cir.ptr_stride %[[NU8]], %[[NOFF]]
// CIR:   %[[NHI:.+]] = cir.cast bitcast %[[NGEP]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   cir.store %arg0, %[[NHI]] : !s64i, !cir.ptr<!s64i>
// LLVM: define dso_local noundef i64 @_Z13takeNuaHiWord9NuaHiWord(i64 %[[NARG:[^,)]+]])
// LLVM:   %[[NSLOT:.+]] = alloca %struct.NuaHiWord, align 8
// LLVM:   %[[NHI:.+]] = getelementptr{{( inbounds)?}} i8, ptr %[[NSLOT]], i64 8
// LLVM:   store i64 %[[NARG]], ptr %[[NHI]], align 8

// The shape libc++ builds std::expected out of.
bool takeNuaExpected(NuaExpected v) { return v.has; }

// CIR: cir.func {{.*}}@_Z15takeNuaExpected11NuaExpected(%arg0: !u64i {{.*}}) -> (!cir.bool
// LLVM: define dso_local noundef zeroext i1 @_Z15takeNuaExpected11NuaExpected(i64 %{{[^,]+}})

NuaExpected retNuaExpected() { return NuaExpected{}; }

// CIR: cir.func {{.*}}@_Z14retNuaExpectedv() -> !u64i
// LLVM: define dso_local i64 @_Z14retNuaExpectedv()

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
