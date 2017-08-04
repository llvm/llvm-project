// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -fcxx-exceptions -emit-pch -o %t.1.ast %S/Inputs/exprs3.cpp
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -fcxx-exceptions -ast-merge %t.1.ast -fsyntax-only -verify %s
// expected-no-diagnostics

static_assert(Ch1 == 'a');
static_assert(Ch2 == 'b');
static_assert(Ch3 == 'c');

static_assert(Ch4 == L'd');
static_assert(Ch5 == L'e');
static_assert(Ch6 == L'f');

static_assert(C1 == 12);
static_assert(C2 == 13);

static_assert(C3 == 12);
static_assert(C4 == 13);

static_assert(C5 == 22L);
static_assert(C6 == 23L);

static_assert(C7 == 66LL);
static_assert(C8 == 67ULL);

static_assert(bval1 == true);
static_assert(bval2 == false);

static_assert(ExpressionTrait == false);

static_assert(ArrayRank == 2);
static_assert(ArrayExtent == 20);

void testImport(int *x, const S1 &cs1, S1 &s1) {
  testNewThrowDelete();
  testArrayElement(nullptr, 12);
  testTernaryOp(0, 1, 2);
  testConstCast(cs1);
  testStaticCast(s1);
  testReinterpretCast(s1);
  testDynamicCast(s1);
  testScalarInit(42);
  testOffsetOf();
  testDefaultArg(12);
  testDefaultArg();
  testDefaultArgExpr();
  useTemplateType();
}
