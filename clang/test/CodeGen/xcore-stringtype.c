// REQUIRES: xcore-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple xcore-unknown-unknown -fno-signed-char -fno-common -emit-llvm -o - %s | FileCheck %s

// CHECK: target triple = "xcore-unknown-unknown"

// In the tests below, some types are not supported by the ABI (_Complex,
// variable length arrays) and will thus emit no meta data.
// The 38 tests that do emit typestrings are gathered into '!xcore.typestrings'
// Please see 'Tools Development Guide' section 2.16.2 for format details:
// <https://www.xmos.com/download/public/Tools-Development-Guide%28X9114A%29.pdf>

// CHECK: !xcore.typestrings = !{
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}},
// CHECK: !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}
// CHECK-NOT: , !{{[0-9]+}}
// CHECK: }


// test BuiltinType
// CHECK: !{{[0-9]+}} = !{void (i1, i8, i8, i8, i16, i16, i16, i32, i32, i32,
// CHECK:      i32, i32, i32, i64, i64, i64, float, double, double)*
// CHECK:      @builtinType, !"f{0}(b,uc,uc,sc,ss,us,ss,si,ui,si,sl,
// CHECK:      ul,sl,sll,ull,sll,ft,d,ld)"}
void builtinType(_Bool B, char C, unsigned char UC, signed char SC, short S,
                 unsigned short US, signed short SS, int I, unsigned int UI,
                 signed int SI, long L, unsigned long UL, signed long SL,
                 long long LL, unsigned long long ULL, signed long long SLL,
                 float F, double D, long double LD) {}
double _Complex Complex; // not supported


// test FunctionType & Qualifiers
// CHECK: !{{[0-9]+}} = !{void ()* @gI, !"f{0}()"}
// CHECK: !{{[0-9]+}} = !{void (...)* @eI, !"f{0}()"}
// CHECK: !{{[0-9]+}} = !{void ()* @gV, !"f{0}(0)"}
// CHECK: !{{[0-9]+}} = !{void ()* @eV, !"f{0}(0)"}
// CHECK: !{{[0-9]+}} = !{void (i32, ...)* @gVA, !"f{0}(si,va)"}
// CHECK: !{{[0-9]+}} = !{void (i32, ...)* @eVA, !"f{0}(si,va)"}
// CHECK: !{{[0-9]+}} = !{i32* (i32*)* @gQ, !"f{crv:p(cv:si)}(p(cv:si))"}
// CHECK: !{{[0-9]+}} = !{i32* (i32*)* @eQ, !"f{crv:p(cv:si)}(p(cv:si))"}
extern void eI();
void gI() {eI();};
extern void eV(void);
void gV(void) {eV();}
extern void eVA(int, ...);
void gVA(int i, ...) {eVA(i);}
extern const volatile int* volatile restrict const
    eQ(const volatile int * volatile restrict const);
const volatile int* volatile restrict const
    gQ(const volatile int * volatile restrict const i) {return eQ(i);}


// test PointerType
// CHECK: !{{[0-9]+}} = !{i32* (i32*, i32* (i32*)*)*
// CHECK:       @pointerType, !"f{p(si)}(p(si),p(f{p(si)}(p(si))))"}
// CHECK: !{{[0-9]+}} = !{i32** @EP, !"p(si)"}
// CHECK: !{{[0-9]+}} = !{i32** @GP, !"p(si)"}
extern int* EP;
int* GP;
int* pointerType(int *I, int * (*FP)(int *)) {
  return I? EP : GP;
}

// test ArrayType
// CHECK: !{{[0-9]+}} = !{[2 x i32]* (i32*, i32*, [2 x i32]*, [2 x i32]*, i32*)*
// CHECK:       @arrayType, !"f{p(a(2:si))}(p(si),p(cv:si),p(a(2:si)),
// CHECK:       p(a(2:si)),p(si))"}
// CHECK: !{{[0-9]+}} = !{[0 x i32]* @EA1, !"a(*:cv:si)"}
// CHECK: !{{[0-9]+}} = !{[2 x i32]* @EA2, !"a(2:si)"}
// CHECK: !{{[0-9]+}} = !{[0 x [2 x i32]]* @EA3, !"a(*:a(2:si))"}
// CHECK: !{{[0-9]+}} = !{[3 x [2 x i32]]* @EA4, !"a(3:a(2:si))"}
// CHECK: !{{[0-9]+}} = !{[2 x i32]* @GA1, !"a(2:cv:si)"}
// CHECK: !{{[0-9]+}} = !{void ([2 x i32]*)* @arrayTypeVariable1,
// CHECK:       !"f{0}(p(a(2:si)))"}
// CHECK: !{{[0-9]+}} = !{void (void ([2 x i32]*)*)* @arrayTypeVariable2,
// CHECK:       !"f{0}(p(f{0}(p(a(2:si)))))"}
// CHECK: !{{[0-9]+}} = !{[3 x [2 x i32]]* @GA2, !"a(3:a(2:si))"}
extern int GA2[3][2];
extern const volatile int EA1[];
extern int EA2[2];
extern int EA3[][2];
extern int EA4[3][2];
const volatile int GA1[2];
extern void arrayTypeVariable1(int[*][2]);
extern void arrayTypeVariable2( void(*fp)(int[*][2]) );
extern void arrayTypeVariable3(int[3][*]);                // not supported
extern void arrayTypeVariable4( void(*fp)(int[3][*]) );   // not supported
typedef int RetType[2];
RetType* arrayType(int A1[], int const volatile A2[2], int A3[][2],
                   int A4[3][2], int A5[const volatile restrict static 2]) {
  if (A1) EA2[0] = EA1[0];
  if (A2) return &EA2;
  if (A3) return EA3;
  if (A4) return EA4;
  if (A5) EA2[0] = GA1[0];
  arrayTypeVariable1(EA4);
  arrayTypeVariable2(arrayTypeVariable1);
  arrayTypeVariable3(EA4);
  arrayTypeVariable4(arrayTypeVariable3);
  return GA2;
}


// test StructureType
// CHECK: !{{[0-9]+}} = !{void (%struct.S1*)* @structureType1,
// CHECK:       !"f{0}(s(S1){m(ps2){p(s(S2){m(ps3){p(s(S3){m(s1){s(S1){}}})}})}})"}
// CHECK: !{{[0-9]+}} = !{void (%struct.S2*)* @structureType2,
// CHECK:       !"f{0}(s(S2){m(ps3){p(s(S3){m(s1){s(S1){m(ps2){p(s(S2){})}}}})}})"}
// CHECK: !{{[0-9]+}} = !{void (%struct.S3*)* @structureType3,
// CHECK:       !"f{0}(s(S3){m(s1){s(S1){m(ps2){p(s(S2){m(ps3){p(s(S3){})}})}}}})"}
// CHECK: !{{[0-9]+}} = !{void (%struct.S4*)* @structureType4,
// CHECK:       !"f{0}(s(S4){m(s1){s(S1){m(ps2){p(s(S2){m(ps3){p(s(S3){m(s1){s(S1){}}})}})}}}})"}
// CHECK: !{{[0-9]+}} = !{%struct.anon* @StructAnon, !"s(){m(A){si}}"}
// CHECK: !{{[0-9]+}} = !{i32 (%struct.SB*)* @structureTypeB,
// CHECK:       !"f{si}(s(SB){m(){b(4:si)},m(){b(2:si)},m(N4){b(4:si)},
// CHECK:       m(N2){b(2:si)},m(){b(4:ui)},m(){b(4:si)},m(){b(4:c:si)},
// CHECK:       m(){b(4:c:si)},m(){b(4:cv:si)}})"}
struct S2;
struct S1{struct S2 *ps2;};
struct S3;
struct S2{struct S3 *ps3;};
struct S3{struct S1 s1;};
struct S4{struct S1 s1;};
void structureType1(struct S1 s1){}
void structureType2(struct S2 s2){}
void structureType3(struct S3 s3){}
void structureType4(struct S4 s4){}
struct {int A;} StructAnon = {1};
struct SB{int:4; int:2; int N4:4; int N2:2; unsigned int:4; signed int:4;
          const int:4; int const :4; volatile const int:4;};
int structureTypeB(struct SB sb){return StructAnon.A;}


// test UnionType
// CHECK: !{{[0-9]+}} = !{void (%union.U1*)* @unionType1,
// CHECK:       !"f{0}(u(U1){m(pu2){p(u(U2){m(pu3){p(u(U3){m(u1){u(U1){}}})}})}})"}
// CHECK: !{{[0-9]+}} = !{void (%union.U2*)* @unionType2,
// CHECK:       !"f{0}(u(U2){m(pu3){p(u(U3){m(u1){u(U1){m(pu2){p(u(U2){})}}}})}})"}
// CHECK: !{{[0-9]+}} = !{void (%union.U3*)* @unionType3,
// CHECK:       !"f{0}(u(U3){m(u1){u(U1){m(pu2){p(u(U2){m(pu3){p(u(U3){})}})}}}})"}
// CHECK: !{{[0-9]+}} = !{void (%union.U4*)* @unionType4,
// CHECK:       !"f{0}(u(U4){m(u1){u(U1){m(pu2){p(u(U2){m(pu3){p(u(U3){m(u1){u(U1){}}})}})}}}})"}
// CHECK: !{{[0-9]+}} = !{%union.anon* @UnionAnon, !"u(){m(A){si}}"}
// CHECK: !{{[0-9]+}} = !{i32 (%union.UB*)* @unionTypeB,
// CHECK:       !"f{si}(u(UB){m(N2){b(2:si)},m(N4){b(4:si)},m(){b(2:si)},
// CHECK:       m(){b(4:c:si)},m(){b(4:c:si)},m(){b(4:cv:si)},m(){b(4:si)},
// CHECK:       m(){b(4:si)},m(){b(4:ui)}})"}
union U2;
union U1{union U2 *pu2;};
union U3;
union U2{union U3 *pu3;};
union U3{union U1 u1;};
union U4{union U1 u1;};
void unionType1(union U1 u1) {}
void unionType2(union U2 u2) {}
void unionType3(union U3 u3) {}
void unionType4(union U4 u4) {}
union UB{int:4; int:2; int N4:4; int N2:2; unsigned int:4; signed int:4;
         const int:4; int const :4; volatile const int:4;};
union {int A;} UnionAnon = {1};
int unionTypeB(union UB ub) {return UnionAnon.A;}


// test EnumType
// CHECK: !{{[0-9]+}} = !{i32* @EnumAnon, !"e(){m(EA){3}}"}
// CHECK: !{{[0-9]+}} = !{i32 (i32)* @enumType,
// CHECK:       !"f{si}(e(E){m(A){7},m(B){6},m(C){5},m(D){0}})"}
enum E {D, C=5, B, A};
enum {EA=3} EnumAnon = EA;
int enumType(enum E e) {return EnumAnon;}


// CHECK: !{{[0-9]+}} = !{i32 ()* @testReDecl, !"f{si}()"}
// CHECK: !{{[0-9]+}} = !{[10 x i32]* @After, !"a(10:si)"}
// CHECK: !{{[0-9]+}} = !{[10 x i32]* @Before, !"a(10:si)"}
extern int After[];
extern int Before[10];
int testReDecl() {return After[0] + Before[0];}
int After[10];
int Before[];

