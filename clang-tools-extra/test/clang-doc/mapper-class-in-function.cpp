// This test requires Linux due to the system-dependent USR for the
// inner class.
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E.bc --dump | FileCheck %s --check-prefix CHECK-H
// RUN: llvm-bcanalyzer %t/docs/bc/01A95F3F73F53281B3E50109A577FD2493159365.bc --dump | FileCheck %s --check-prefix CHECK-H-I

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
  // CHECK-H-NEXT: <DefLocation abbrevid=7 op0=12 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-H-NEXT: <TypeBlock NumWords=4 BlockCodeSize=4>
    // CHECK-H-NEXT: <Type abbrevid=4 op0=4 op1=4/> blob data = 'void'
  // CHECK-H-NEXT: </TypeBlock>
// CHECK-H-NEXT: </FunctionBlock>

// CHECK-H-I: <BLOCKINFO_BLOCK/>
// CHECK-H-I-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-H-I-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-H-I-NEXT: </VersionBlock>
// CHECK-H-I-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-H-I-NEXT: <USR abbrevid=4 op0=20 op1=1 op2=169 op3=95 op4=63 op5=115 op6=245 op7=50 op8=129 op9=179 op10=229 op11=1 op12=9 op13=165 op14=119 op15=253 op16=36 op17=147 op18=21 op19=147 op20=101/>
  // CHECK-H-I-NEXT: <Name abbrevid=5 op0=1/> blob data = 'I'
  // CHECK-H-I-NEXT: <Namespace abbrevid=6 op0=2 op1=40/> blob data = 'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
  // CHECK-H-I-NEXT: <DefLocation abbrevid=7 op0=13 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-H-I-NEXT: <TagType abbrevid=9 op0=3/>
// CHECK-H-I-NEXT: </RecordBlock>


