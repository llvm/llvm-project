// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/289584A8E0FF4178A794622A547AA622503967A1.bc --dump | FileCheck %s

class E {};

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
  // CHECK-NEXT: <DefLocation abbrevid=6 op0=8 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-NEXT: </RecordBlock>
