// RUN: %clang_cc1 -x c++ -Wno-return-type -fsycl-is-device -std=c++11 -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

//CHECK: FunctionDecl{{.*}}foo1
void foo1()
{
  //CHECK: VarDecl{{.*}}v_one
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  __attribute__((__doublepump__))
  unsigned int v_one[64];

  //CHECK: VarDecl{{.*}}v_one2
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::doublepump]] unsigned int v_one2[64];

  //CHECK: VarDecl{{.*}}v_two
  //CHECK: IntelFPGAMemoryAttr
  __attribute__((__memory__))
  unsigned int v_two[64];

  //CHECK: VarDecl{{.*}}v_two2
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB
  [[intelfpga::memory("MLAB")]] unsigned int v_two2[64];

  //CHECK: VarDecl{{.*}}v_two3
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM
  [[intelfpga::memory("BLOCK_RAM")]] unsigned int v_two3[32];

  //CHECK: VarDecl{{.*}}v_three
  //CHECK: IntelFPGARegisterAttr
  __attribute__((__register__))
  unsigned int v_three[64];

  //CHECK: VarDecl{{.*}}v_three2
  //CHECK: IntelFPGARegisterAttr
  [[intelfpga::register]] unsigned int v_three2[32];

  //CHECK: VarDecl{{.*}}v_four
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  __attribute__((__singlepump__))
  unsigned int v_four[64];

  //CHECK: VarDecl{{.*}}v_four2
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intelfpga::singlepump]] unsigned int v_four2[64];

  //CHECK: VarDecl{{.*}}v_five
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  __attribute__((__bankwidth__(4)))
  unsigned int v_five[64];

  //CHECK: VarDecl{{.*}}v_five2
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::bankwidth(8)]] unsigned int v_five2[32];

  //CHECK: VarDecl{{.*}}v_six
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  __attribute__((__numbanks__(8)))
  unsigned int v_six[64];

  //CHECK: VarDecl{{.*}}v_six2
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::numbanks(4)]] unsigned int v_six2[32];

  //CHECK: VarDecl{{.*}}v_seven
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxConcurrencyAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  __attribute__((max_concurrency(4)))
  unsigned int v_seven[64];

  //CHECK: VarDecl{{.*}}v_seven2
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxConcurrencyAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::max_concurrency(8)]] unsigned int v_seven2[64];

  //CHECK: VarDecl{{.*}}v_fourteen
  //CHECK: IntelFPGADoublePumpAttr
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  __attribute__((__doublepump__))
  __attribute__((__memory__("MLAB")))
  unsigned int v_fourteen[64];

  //CHECK: VarDecl{{.*}}v_fifteen
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  __attribute__((__memory__("MLAB")))
  __attribute__((__doublepump__))
  unsigned int v_fifteen[64];

  int __attribute__((__register__)) A;
  int __attribute__((__numbanks__(4), __bankwidth__(16), __singlepump__)) B;
  int __attribute__((__numbanks__(4), __bankwidth__(16), __doublepump__)) C;
  int __attribute__((__numbanks__(4), __bankwidth__(16))) E;

  // diagnostics

  // **doublepump
  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__doublepump__))
  __attribute__((__singlepump__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dp_one[64];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  __attribute__((doublepump))
  __attribute__((__doublepump__))
  unsigned int dp_two[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__doublepump__))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dp_three[64];

  // **singlepump
  //expected-error@+1{{attributes are not compatible}}
  __attribute__((__singlepump__,__doublepump__))
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int sp_one[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  __attribute__((singlepump))
  __attribute__((__singlepump__))
  unsigned int sp_two[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__singlepump__))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int sp_three[64];

  // **register
  //expected-warning@+1{{attribute 'register' is already applied}}
  __attribute__((register)) __attribute__((__register__))
  unsigned int reg_one[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__singlepump__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_two[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__doublepump__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_three[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__memory__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_four[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__bankwidth__(16)))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_six[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__max_concurrency__(16)))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_six_two[64];

  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__register__))
  __attribute__((__numbanks__(8)))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_seven[64];

  // **memory
  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__memory__))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_one[64];

  //expected-warning@+1{{attribute 'memory' is already applied}}
  __attribute__((memory)) __attribute__((__memory__))
  unsigned int mem_two[64];

  // bankwidth
  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__bankwidth__(16)))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int bw_one[64];

  //CHECK: VarDecl{{.*}}bw_two
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'bankwidth' is already applied}}
  __attribute__((__bankwidth__(8)))
  __attribute__((__bankwidth__(16)))
  unsigned int bw_two[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  __attribute__((__bankwidth__(3)))
  unsigned int bw_three[64];

  //expected-error@+1{{'bankwidth' attribute requires integer constant between 1 and 1048576 inclusive}}
  __attribute__((__bankwidth__(-4)))
  unsigned int bw_four[64];

  int i_bankwidth = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  __attribute__((__bankwidth__(i_bankwidth)))
  //expected-note@-1{{read of non-const variable 'i_bankwidth' is not allowed in a constant expression}}
  unsigned int bw_five[64];

  //expected-error@+1{{'__bankwidth__' attribute takes one argument}}
  __attribute__((__bankwidth__(4,8)))
  unsigned int bw_six[64];

  //expected-error@+1{{'bankwidth' attribute requires integer constant between 1 and 1048576 inclusive}}
  __attribute__((__bankwidth__(0)))
  unsigned int bw_seven[64];

  // max_concurrency
  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__max_concurrency__(16)))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mc_one[64];

  //CHECK: VarDecl{{.*}}mc_two
  //CHECK: IntelFPGAMaxConcurrencyAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGAMaxConcurrencyAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{is already applied}}
  __attribute__((__max_concurrency__(8)))
  __attribute__((__max_concurrency__(16)))
  unsigned int mc_two[64];

  //expected-error@+1{{'max_concurrency' attribute requires integer constant between 0 and 1048576 inclusive}}
  __attribute__((__max_concurrency__(-4)))
  unsigned int mc_four[64];

  int i_max_concurrency = 32; // expected-note {{declared here}}
  //expected-error@+1{{expression is not an integral constant expression}}
  __attribute__((__max_concurrency__(i_max_concurrency)))
  //expected-note@-1{{read of non-const variable 'i_max_concurrency' is not allowed in a constant expression}}
  unsigned int mc_five[64];

  //expected-error@+1{{'__max_concurrency__' attribute takes one argument}}
  __attribute__((__max_concurrency__(4,8)))
  unsigned int mc_six[64];

  // numbanks
  //expected-error@+2{{attributes are not compatible}}
  __attribute__((__numbanks__(16)))
  __attribute__((__register__))
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int nb_one[64];

  //CHECK: VarDecl{{.*}}nb_two
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{attribute 'numbanks' is already applied}}
  __attribute__((__numbanks__(8)))
  __attribute__((__numbanks__(16)))
  unsigned int nb_two[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  __attribute__((__numbanks__(15)))
  unsigned int nb_three[64];

  //expected-error@+1{{attribute requires integer constant between 1 and 1048576 inclusive}}
  __attribute__((__numbanks__(-4)))
  unsigned int nb_four[64];

  int i_numbanks = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  __attribute__((__numbanks__(i_numbanks)))
  //expected-note@-1{{read of non-const variable 'i_numbanks' is not allowed in a constant expression}}
  unsigned int nb_five[64];

  //expected-error@+1{{'__numbanks__' attribute takes one argument}}
  __attribute__((__numbanks__(4,8)))
  unsigned int nb_six[64];

  //expected-error@+1{{'numbanks' attribute requires integer constant between 1 and 1048576 inclusive}}
  __attribute__((__numbanks__(0)))
  unsigned int nb_seven[64];
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
__attribute__((__max_concurrency__(8)))
__constant unsigned int ext_two[64] = { 1, 2, 3 };

void other2()
{
  //expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
  __attribute__((__max_concurrency__(8))) const int ext_six[64] = { 0, 1 };
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
void other3(__attribute__((__max_concurrency__(8))) int pfoo) {}

struct foo {
  //CHECK: FieldDecl{{.*}}v_one
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  __attribute__((__doublepump__)) unsigned int v_one[64];

  //CHECK: FieldDecl{{.*}}v_two
  //CHECK: IntelFPGAMemoryAttr
  __attribute__((__memory__)) unsigned int v_two[64];

  //CHECK: FieldDecl{{.*}}v_two_A
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  __attribute__((__memory__("MLAB"))) unsigned int v_two_A[64];

  //CHECK: FieldDecl{{.*}}v_two_B
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  __attribute__((__memory__("BLOCK_RAM"))) unsigned int v_two_B[64];

  //CHECK: FieldDecl{{.*}}v_two_C
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  __attribute__((__memory__("BLOCK_RAM")))
  __attribute__((doublepump)) unsigned int v_two_C[64];

  //CHECK: FieldDecl{{.*}}v_three
  //CHECK: IntelFPGARegisterAttr
  __attribute__((__register__)) unsigned int v_three[64];

  //CHECK: FieldDecl{{.*}}v_four
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  __attribute__((__singlepump__)) unsigned int v_four[64];

  //CHECK: FieldDecl{{.*}}v_five
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  __attribute__((__bankwidth__(4))) unsigned int v_five[64];

  //CHECK: FieldDecl{{.*}}v_six
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  __attribute__((__numbanks__(8))) unsigned int v_six[64];

  //CHECK: FieldDecl{{.*}}v_seven
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxConcurrencyAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  __attribute__((__max_concurrency__(4))) unsigned int v_seven[64];
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo1();
  });
  return 0;
}
