// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/A44B32CC3C087C9AF75DAF50DE193E85E7B2C16B.bc --dump | FileCheck %s

int F(int param) { return param; }

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=164 op2=75 op3=50 op4=204 op5=60 op6=8 op7=124 op8=154 op9=247 op10=93 op11=175 op12=80 op13=222 op14=25 op15=62 op16=133 op17=231 op18=178 op19=193 op20=107/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'F'
  // CHECK-NEXT: <DefLocation abbrevid=7 op0=8 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <TypeBlock NumWords=4 BlockCodeSize=4>
    // CHECK-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
  // CHECK-NEXT: </TypeBlock>
  // CHECK-NEXT: <FieldTypeBlock NumWords=7 BlockCodeSize=4>
    // CHECK-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
    // CHECK-NEXT: <Name abbrevid=5 op0=5/> blob data = 'param'
  // CHECK-NEXT: </FieldTypeBlock>
// CHECK-NEXT: </FunctionBlock>
