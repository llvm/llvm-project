// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/8D042EFFC98B373450BC6B5B90A330C25A150E9C.bc --dump | FileCheck %s

namespace A {}

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <NamespaceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-NEXT: </NamespaceBlock>
