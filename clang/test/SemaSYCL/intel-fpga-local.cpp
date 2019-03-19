// RUN: %clang_cc1 -x c++ -Wno-return-type -fsycl-is-device -std=c++11 -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

//CHECK: FunctionDecl{{.*}}foo1
void foo1()
{
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
  __attribute__((__numbanks__(4))) unsigned int v_six2[32];

  int __attribute__((__register__)) A;
  int __attribute__((__numbanks__(4), __bankwidth__(16))) E;

  // diagnostics

  // **register
  //expected-warning@+1{{attribute 'register' is already applied}}
  __attribute__((register)) __attribute__((__register__))
  unsigned int reg_one[64];

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

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
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

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  __attribute__((__bankwidth__(0)))
  unsigned int bw_seven[64];

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

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
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

  //expected-error@+1{{requires integer constant between 1 and 1048576}}
  __attribute__((__numbanks__(0)))
  unsigned int nb_seven[64];
}

struct foo {
  //CHECK: FieldDecl{{.*}}v_two
  //CHECK: IntelFPGAMemoryAttr
  __attribute__((__memory__)) unsigned int v_two[64];

  //CHECK: FieldDecl{{.*}}v_two_A
  //CHECK: IntelFPGAMemoryAttr{{.*}}MLAB{{$}}
  __attribute__((__memory__("MLAB"))) unsigned int v_two_A[64];

  //CHECK: FieldDecl{{.*}}v_two_B
  //CHECK: IntelFPGAMemoryAttr{{.*}}BlockRAM{{$}}
  __attribute__((__memory__("BLOCK_RAM"))) unsigned int v_two_B[64];

  //CHECK: FieldDecl{{.*}}v_three
  //CHECK: IntelFPGARegisterAttr
  __attribute__((__register__)) unsigned int v_three[64];

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
