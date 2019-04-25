// RUN: %clang_cc1 -x c++ -Wno-return-type -fsycl-is-device -std=c++11 -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

//CHECK: FunctionDecl{{.*}}foo1
void foo1()
{
  //CHECK: VarDecl{{.*}}v_one
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::doublepump]] unsigned int v_one[64];

  //CHECK: VarDecl{{.*}}v_two
  //CHECK: IntelFPGAMemoryAttr
  [[intelfpga::memory]] unsigned int v_two[64];

  //CHECK: VarDecl{{.*}}v_two2
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB
  [[intelfpga::memory("MLAB")]] unsigned int v_two2[64];

  //CHECK: VarDecl{{.*}}v_two3
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM
  [[intelfpga::memory("BLOCK_RAM")]] unsigned int v_two3[32];

  //CHECK: VarDecl{{.*}}v_three
  //CHECK: IntelFPGARegisterAttr
  [[intelfpga::register]] unsigned int v_three[64];

  //CHECK: VarDecl{{.*}}v_four
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intelfpga::singlepump]] unsigned int v_four[64];

  //CHECK: VarDecl{{.*}}v_five
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::bankwidth(4)]] unsigned int v_five[32];

  //CHECK: VarDecl{{.*}}v_six
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::numbanks(8)]] unsigned int v_six[32];

  //CHECK: VarDecl{{.*}}v_seven
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::max_private_copies(8)]] unsigned int v_seven[64];

  //CHECK: VarDecl{{.*}}v_fourteen
  //CHECK: IntelFPGADoublePumpAttr
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intelfpga::doublepump]]
  [[intelfpga::memory("MLAB")]]
  unsigned int v_fourteen[64];

  //CHECK: VarDecl{{.*}}v_fifteen
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::memory("MLAB")]]
  [[intelfpga::doublepump]]
  unsigned int v_fifteen[64];

  [[intelfpga::register]] int A;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16), intelfpga::singlepump]] int B;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16), intelfpga::doublepump]] int C;
  [[intelfpga::numbanks(4), intelfpga::bankwidth(16)]] int E;

  // diagnostics

  // **doublepump
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::doublepump]]
  [[intelfpga::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dp_one[64];

  //expected-warning@+2{{attribute 'doublepump' is already applied}}
  [[intelfpga::doublepump]]
  [[intelfpga::doublepump]]
  unsigned int dp_two[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::doublepump]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int dp_three[64];

  // **singlepump
  //expected-error@+1{{attributes are not compatible}}
  [[intelfpga::singlepump, intelfpga::doublepump]]
  //expected-note@-1 {{conflicting attribute is here}}
  unsigned int sp_one[64];

  //expected-warning@+2{{attribute 'singlepump' is already applied}}
  [[intelfpga::singlepump]]
  [[intelfpga::singlepump]]
  unsigned int sp_two[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::singlepump]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int sp_three[64];

  // **register
  //expected-warning@+1{{attribute 'register' is already applied}}
  [[intelfpga::register]] [[intelfpga::register]]
  unsigned int reg_one[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::singlepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_two[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::doublepump]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_three[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::memory]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_four[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::bankwidth(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_six[64];

  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::max_private_copies(16)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_six_two[64];


  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::register]]
  [[intelfpga::numbanks(8)]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int reg_seven[64];

  // **memory
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::memory]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mem_one[64];

  //expected-warning@+1{{attribute 'memory' is already applied}}
  [[intelfpga::memory]] [[intelfpga::memory]]
  unsigned int mem_two[64];

  // bankwidth
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::bankwidth(16)]]
  [[intelfpga::register]]
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
  [[intelfpga::bankwidth(8)]]
  [[intelfpga::bankwidth(16)]]
  unsigned int bw_two[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intelfpga::bankwidth(3)]]
  unsigned int bw_three[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::bankwidth(-4)]]
  unsigned int bw_four[64];

  int i_bankwidth = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intelfpga::bankwidth(i_bankwidth)]]
  //expected-note@-1{{read of non-const variable 'i_bankwidth' is not allowed in a constant expression}}
  unsigned int bw_five[64];

  //expected-error@+1{{'bankwidth' attribute takes one argument}}
  [[intelfpga::bankwidth(4,8)]]
  unsigned int bw_six[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::bankwidth(0)]]
  unsigned int bw_seven[64];


  // max_private_copies_
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::max_private_copies(16)]]
  [[intelfpga::register]]
  //expected-note@-2 {{conflicting attribute is here}}
  unsigned int mc_one[64];

  //CHECK: VarDecl{{.*}}mc_two
  //CHECK: IntelFPGAMaxPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  //CHECK: IntelFPGAMaxPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
  //expected-warning@+2{{is already applied}}
  [[intelfpga::max_private_copies(8)]]
  [[intelfpga::max_private_copies(16)]]
  unsigned int mc_two[64];

  //expected-error@+1{{'max_private_copies' attribute requires integer constant between 0 and 1048576 inclusive}}
  [[intelfpga::max_private_copies(-4)]]
  unsigned int mc_four[64];

  int i_max_private_copies = 32; // expected-note {{declared here}}
  //expected-error@+1{{expression is not an integral constant expression}}
  [[intelfpga::max_private_copies(i_max_private_copies)]]
  //expected-note@-1{{read of non-const variable 'i_max_private_copies' is not allowed in a constant expression}}
  unsigned int mc_five[64];

  //expected-error@+1{{'max_private_copies' attribute takes one argument}}
  [[intelfpga::max_private_copies(4,8)]]
  unsigned int mc_six[64];

  // numbanks
  //expected-error@+2{{attributes are not compatible}}
  [[intelfpga::numbanks(16)]]
  [[intelfpga::register]]
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
  [[intelfpga::numbanks(8)]]
  [[intelfpga::numbanks(16)]]
  unsigned int nb_two[64];

  //expected-error@+1{{must be a constant power of two greater than zero}}
  [[intelfpga::numbanks(15)]]
  unsigned int nb_three[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::numbanks(-4)]]
  unsigned int nb_four[64];

  int i_numbanks = 32; // expected-note {{declared here}}
  //expected-error@+1{{is not an integral constant expression}}
  [[intelfpga::numbanks(i_numbanks)]]
  //expected-note@-1{{read of non-const variable 'i_numbanks' is not allowed in a constant expression}}
  unsigned int nb_five[64];

  //expected-error@+1{{'numbanks' attribute takes one argument}}
  [[intelfpga::numbanks(4,8)]]
  unsigned int nb_six[64];

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  [[intelfpga::numbanks(0)]]
  unsigned int nb_seven[64];

  // GNU style
  //expected-warning@+1{{unknown attribute 'numbanks' ignored}}
  int __attribute__((numbanks(4))) a_one;

  //expected-warning@+1{{unknown attribute 'memory' ignored}}
  unsigned int __attribute__((memory("MLAB"))) a_two;

  //expected-warning@+1{{unknown attribute 'bankwidth' ignored}}
  int __attribute__((bankwidth(8))) a_three;

  //expected-warning@+1{{unknown attribute 'register' ignored}}
  int __attribute__((register)) a_four;

  //expected-warning@+1{{unknown attribute '__singlepump__' ignored}}
  unsigned int __attribute__((__singlepump__)) a_five;

  //expected-warning@+1{{unknown attribute '__doublepump__' ignored}}
  unsigned int __attribute__((__doublepump__)) a_six;

  //expected-warning@+1{{unknown attribute '__max_private_copies__' ignored}}
  int __attribute__((__max_private_copies__(4))) a_seven;
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
[[intelfpga::max_private_copies(8)]]
__constant unsigned int ext_two[64] = { 1, 2, 3 };

void other2()
{
  //expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
  [[intelfpga::max_private_copies(8)]] const int ext_six[64] = { 0, 1 };
}

//expected-error@+1{{attribute only applies to local non-const variables and non-static data members}}
void other3([[intelfpga::max_private_copies(8)]] int pfoo) {}

struct foo {
  //CHECK: FieldDecl{{.*}}v_one
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::doublepump]] unsigned int v_one[64];

  //CHECK: FieldDecl{{.*}}v_two
  //CHECK: IntelFPGAMemoryAttr
  [[intelfpga::memory]] unsigned int v_two[64];

  //CHECK: FieldDecl{{.*}}v_two_A
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  [[intelfpga::memory("MLAB")]] unsigned int v_two_A[64];

  //CHECK: FieldDecl{{.*}}v_two_B
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  [[intelfpga::memory("BLOCK_RAM")]] unsigned int v_two_B[64];

  //CHECK: FieldDecl{{.*}}v_two_C
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  //CHECK: IntelFPGADoublePumpAttr
  [[intelfpga::memory("BLOCK_RAM")]]
  [[intelfpga::doublepump]] unsigned int v_two_C[64];

  //CHECK: FieldDecl{{.*}}v_three
  //CHECK: IntelFPGARegisterAttr
  [[intelfpga::register]] unsigned int v_three[64];

  //CHECK: FieldDecl{{.*}}v_four
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGASinglePumpAttr
  [[intelfpga::singlepump]] unsigned int v_four[64];

  //CHECK: FieldDecl{{.*}}v_five
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGABankWidthAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::bankwidth(4)]] unsigned int v_five[64];

  //CHECK: FieldDecl{{.*}}v_six
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGANumBanksAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  [[intelfpga::numbanks(8)]] unsigned int v_six[64];

  //CHECK: FieldDecl{{.*}}v_seven
  //CHECK: IntelFPGAMemoryAttr{{.*}}Implicit
  //CHECK: IntelFPGAMaxPrivateCopiesAttr
  //CHECK-NEXT: ConstantExpr
  //CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  [[intelfpga::max_private_copies(4)]] unsigned int v_seven[64];
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
