// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E.bc --dump | FileCheck %s --check-prefix CHECK-H
// RUN: llvm-bcanalyzer %t/docs/bc/E03E804368784360D86C757B549D14BB84A94415.bc --dump | FileCheck %s --check-prefix CHECK-H-I

void H() {
  class I {};
}

// CHECK-H: <BLOCKINFO_BLOCK/>
// CHECK-H-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-H-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-H-NEXT: </VersionBlock>
// CHECK-H-NEXT: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-H-NEXT: <USR abbrevid=4 op0=20 op1=182 op2=172 op3=76 op4=92 op5=159 op6=46 op7=163 op8=242 op9=179 op10=236 op11=225 op12=163 op13=61 op14=52 op15=159 op16=78 op17=229 op18=2 op19=178 op20=78/>
  // CHECK-H-NEXT: <Name abbrevid=5 op0=1/> blob data = 'H'
  // CHECK-H-NEXT: <DefLocation abbrevid=7 op0=9 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-H-NEXT: <TypeBlock NumWords=4 BlockCodeSize=4>
    // CHECK-H-NEXT: <Type abbrevid=4 op0=4 op1=4/> blob data = 'void'
  // CHECK-H-NEXT: </TypeBlock>
// CHECK-H-NEXT: </FunctionBlock>

// CHECK-H-I: <BLOCKINFO_BLOCK/>
// CHECK-H-I-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-H-I-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-H-I-NEXT: </VersionBlock>
// CHECK-H-I-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-H-I-NEXT: <USR abbrevid=4 op0=20 op1=224 op2=62 op3=128 op4=67 op5=104 op6=120 op7=67 op8=96 op9=216 op10=108 op11=117 op12=123 op13=84 op14=157 op15=20 op16=187 op17=132 op18=169 op19=68 op20=21/>
  // CHECK-H-I-NEXT: <Name abbrevid=5 op0=1/> blob data = 'I'
  // CHECK-H-I-NEXT: <Namespace abbrevid=6 op0=2 op1=40/> blob data = 'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
  // CHECK-H-I-NEXT: <DefLocation abbrevid=7 op0=10 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-H-I-NEXT: <TagType abbrevid=9 op0=3/>
// CHECK-H-I-NEXT: </RecordBlock>


