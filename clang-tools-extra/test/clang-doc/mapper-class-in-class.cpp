// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/641AB4A3D36399954ACDE29C7A8833032BF40472.bc --dump | FileCheck %s --check-prefix CHECK-X-Y
// RUN: llvm-bcanalyzer %t/docs/bc/CA7C7935730B5EACD25F080E9C83FA087CCDC75E.bc --dump | FileCheck %s --check-prefix CHECK-X

class X {
  class Y {};
};

// CHECK-X: <BLOCKINFO_BLOCK/>
// CHECK-X-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-X-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-X-NEXT: </VersionBlock>
// CHECK-X-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-X-NEXT: <USR abbrevid=4 op0=20 op1=202 op2=124 op3=121 op4=53 op5=115 op6=11 op7=94 op8=172 op9=210 op10=95 op11=8 op12=14 op13=156 op14=131 op15=250 op16=8 op17=124 op18=205 op19=199 op20=94/>
  // CHECK-X-NEXT: <Name abbrevid=5 op0=1/> blob data = 'X'
  // CHECK-X-NEXT: <DefLocation abbrevid=7 op0=9 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-X-NEXT: <TagType abbrevid=9 op0=3/>
// CHECK-X-NEXT: </RecordBlock>


// CHECK-X-Y: <BLOCKINFO_BLOCK/>
// CHECK-X-Y-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-X-Y-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-X-Y-NEXT: </VersionBlock>
// CHECK-X-Y-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-X-Y-NEXT: <USR abbrevid=4 op0=20 op1=100 op2=26 op3=180 op4=163 op5=211 op6=99 op7=153 op8=149 op9=74 op10=205 op11=226 op12=156 op13=122 op14=136 op15=51 op16=3 op17=43 op18=244 op19=4 op20=114/>
  // CHECK-X-Y-NEXT: <Name abbrevid=5 op0=1/> blob data = 'Y'
  // CHECK-X-Y-NEXT: <Namespace abbrevid=6 op0=1 op1=40/> blob data = 'CA7C7935730B5EACD25F080E9C83FA087CCDC75E'
  // CHECK-X-Y-NEXT: <DefLocation abbrevid=7 op0=10 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-X-Y-NEXT: <TagType abbrevid=9 op0=3/>
// CHECK-X-Y-NEXT: </RecordBlock>
