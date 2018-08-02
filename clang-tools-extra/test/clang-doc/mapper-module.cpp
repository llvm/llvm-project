// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

export module M;

int moduleFunction(int x); // ModuleLinkage

static int staticModuleFunction(int x); // ModuleInternalLinkage

export double exportedModuleFunction(double y, int z); // ExternalLinkage

// RUN: clang-doc --dump-mapper --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/0000000000000000000000000000000000000000.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <NamespaceBlock NumWords=50 BlockCodeSize=4>
// CHECK-0-NEXT:   <FunctionBlock NumWords=47 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=22/> blob data = 'exportedModuleFunction'
// CHECK-0-NEXT:     <Location abbrevid=7 op0=15 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <TypeBlock NumWords=7 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=4 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=6/> blob data = 'double'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:     </TypeBlock>
// CHECK-0-NEXT:     <FieldTypeBlock NumWords=9 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=4 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=6/> blob data = 'double'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'y'
// CHECK-0-NEXT:     </FieldTypeBlock>
// CHECK-0-NEXT:     <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'z'
// CHECK-0-NEXT:     </FieldTypeBlock>
// CHECK-0-NEXT:   </FunctionBlock>
// CHECK-0-NEXT: </NamespaceBlock>
