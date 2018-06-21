// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/06B5F6A19BA9F6A832E127C9968282B94619B210.bc --dump | FileCheck %s

struct C { int i; };

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=6 op2=181 op3=246 op4=161 op5=155 op6=169 op7=246 op8=168 op9=50 op10=225 op11=39 op12=201 op13=150 op14=130 op15=130 op16=185 op17=70 op18=25 op19=178 op20=16/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'C'
  // CHECK-NEXT: <DefLocation abbrevid=6 op0=8 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <MemberTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-NEXT: </ReferenceBlock>
    // CHECK-NEXT: <Name abbrevid=4 op0=1/> blob data = 'i'
  // CHECK-NEXT: </MemberTypeBlock>
// CHECK-NEXT: </RecordBlock>
