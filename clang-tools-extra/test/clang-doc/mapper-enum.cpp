// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/FC07BD34D5E77782C263FA944447929EA8753740.bc --dump | FileCheck %s --check-prefix CHECK-B
// RUN: llvm-bcanalyzer %t/docs/bc/020E6C32A700C3170C009FCCD41671EDDBEAF575.bc --dump | FileCheck %s --check-prefix CHECK-C

enum B { X, Y };

// CHECK-B: <BLOCKINFO_BLOCK/>
// CHECK-B-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-B-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-B-NEXT: </VersionBlock>
// CHECK-B-NEXT: <EnumBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-B-NEXT: <USR abbrevid=4 op0=20 op1=252 op2=7 op3=189 op4=52 op5=213 op6=231 op7=119 op8=130 op9=194 op10=99 op11=250 op12=148 op13=68 op14=71 op15=146 op16=158 op17=168 op18=117 op19=55 op20=64/>
  // CHECK-B-NEXT: <Name abbrevid=5 op0=1/> blob data = 'B'
  // CHECK-B-NEXT: <DefLocation abbrevid=6 op0=9 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-B-NEXT: <Member abbrevid=8 op0=1/> blob data = 'X'
  // CHECK-B-NEXT: <Member abbrevid=8 op0=1/> blob data = 'Y'
// CHECK-B-NEXT: </EnumBlock>

enum class C { A, B };

// CHECK-C: <BLOCKINFO_BLOCK/>
// CHECK-C-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-C-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-C-NEXT: </VersionBlock>
// CHECK-C-NEXT: <EnumBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-C-NEXT: <USR abbrevid=4 op0=20 op1=2 op2=14 op3=108 op4=50 op5=167 op6=0 op7=195 op8=23 op9=12 op10=0 op11=159 op12=204 op13=212 op14=22 op15=113 op16=237 op17=219 op18=234 op19=245 op20=117/>
  // CHECK-C-NEXT: <Name abbrevid=5 op0=1/> blob data = 'C'
  // CHECK-C-NEXT: <DefLocation abbrevid=6 op0=23 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-C-NEXT: <Scoped abbrevid=9 op0=1/>
  // CHECK-C-NEXT: <Member abbrevid=8 op0=1/> blob data = 'A'
  // CHECK-C-NEXT: <Member abbrevid=8 op0=1/> blob data = 'B'
// CHECK-C-NEXT: </EnumBlock>
