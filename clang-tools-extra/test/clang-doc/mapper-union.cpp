// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/0B8A6B938B939B77C6325CCCC8AA3E938BF9E2E8.bc --dump | FileCheck %s

union D { int X; int Y; };

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=11 op2=138 op3=107 op4=147 op5=139 op6=147 op7=155 op8=119 op9=198 op10=50 op11=92 op12=204 op13=200 op14=170 op15=62 op16=147 op17=139 op18=249 op19=226 op20=232/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'D'
  // CHECK-NEXT: <DefLocation abbrevid=7 op0=8 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <TagType abbrevid=9 op0=2/>
  // CHECK-NEXT: <MemberTypeBlock NumWords=6 BlockCodeSize=4>
    // CHECK-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
    // CHECK-NEXT: <Name abbrevid=5 op0=4/> blob data = 'D::X'
    // CHECK-NEXT: <Access abbrevid=6 op0=3/>
  // CHECK-NEXT: </MemberTypeBlock>
  // CHECK-NEXT: <MemberTypeBlock NumWords=6 BlockCodeSize=4>
    // CHECK-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
    // CHECK-NEXT: <Name abbrevid=5 op0=4/> blob data = 'D::Y'
    // CHECK-NEXT: <Access abbrevid=6 op0=3/>
  // CHECK-NEXT: </MemberTypeBlock>
// CHECK-NEXT: </RecordBlock>
