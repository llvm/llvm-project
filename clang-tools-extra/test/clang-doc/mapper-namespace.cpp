// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

namespace A {
  
void f();

}  // namespace A

namespace A {

void f(){};

namespace B {

enum E { X };

E func(int i) { return X; }

}  // namespace B
}  // namespace A

// RUN: clang-doc --dump-mapper --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/8D042EFFC98B373450BC6B5B90A330C25A150E9C.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <NamespaceBlock NumWords=40 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:   <FunctionBlock NumWords=32 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'f'
// CHECK-0-NEXT:     <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <DefLocation abbrevid=6 op0=17 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:     </TypeBlock>
// CHECK-0-NEXT:   </FunctionBlock>
// CHECK-0-NEXT: </NamespaceBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/E21AF79E2A9D02554BA090D10DF39FE273F5CDB5.bc | FileCheck %s --check-prefix CHECK-1
// CHECK-1: <BLOCKINFO_BLOCK/>
// CHECK-1-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-1-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-1-NEXT: </VersionBlock>
// CHECK-1-NEXT: <NamespaceBlock NumWords=64 BlockCodeSize=4>
// CHECK-1-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:   <FunctionBlock NumWords=56 BlockCodeSize=4>
// CHECK-1-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:     <Name abbrevid=5 op0=4/> blob data = 'func'
// CHECK-1-NEXT:     <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-1-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:       <Name abbrevid=5 op0=1/> blob data = 'B'
// CHECK-1-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-1-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-1-NEXT:     </ReferenceBlock>
// CHECK-1-NEXT:     <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-1-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:       <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-1-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-1-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-1-NEXT:     </ReferenceBlock>
// CHECK-1-NEXT:     <DefLocation abbrevid=6 op0=23 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:     <TypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-1-NEXT:       <ReferenceBlock NumWords=5 BlockCodeSize=4>
// CHECK-1-NEXT:         <Name abbrevid=5 op0=12/> blob data = 'enum A::B::E'
// CHECK-1-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-1-NEXT:       </ReferenceBlock>
// CHECK-1-NEXT:     </TypeBlock>
// CHECK-1-NEXT:     <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-1-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-1-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-1-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-1-NEXT:       </ReferenceBlock>
// CHECK-1-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'i'
// CHECK-1-NEXT:     </FieldTypeBlock>
// CHECK-1-NEXT:   </FunctionBlock>
// CHECK-1-NEXT: </NamespaceBlock>
