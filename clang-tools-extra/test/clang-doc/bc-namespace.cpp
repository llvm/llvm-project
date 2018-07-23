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

// RUN: clang-doc --dump-intermediate --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/E9ABF7E7E2425B626723D41E76E4BC7E7A5BD775.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <EnumBlock NumWords=38 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1=233 op2=171 op3=247 op4=231 op5=226 op6=66 op7=91 op8=98 op9=103 op10=35 op11=212 op12=30 op13=118 op14=228 op15=188 op16=126 op17=122 op18=91 op19=215 op20=117/>
// CHECK-0-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-0-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'B'
// CHECK-0-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-0-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:   </ReferenceBlock>
// CHECK-0-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-0-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-0-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:   </ReferenceBlock>
// CHECK-0-NEXT:   <DefLocation abbrevid=6 op0=21 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:   <Member abbrevid=8 op0=1/> blob data = 'X'
// CHECK-0-NEXT: </EnumBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/39D3C95A5F7CE2BA4937BD7B01BAE09EBC2AD8AC.bc | FileCheck %s --check-prefix CHECK-1
// CHECK-1: <BLOCKINFO_BLOCK/>
// CHECK-1-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-1-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-1-NEXT: </VersionBlock>
// CHECK-1-NEXT: <FunctionBlock NumWords=35 BlockCodeSize=4>
// CHECK-1-NEXT:   <USR abbrevid=4 op0=20 op1=57 op2=211 op3=201 op4=90 op5=95 op6=124 op7=226 op8=186 op9=73 op10=55 op11=189 op12=123 op13=1 op14=186 op15=224 op16=158 op17=188 op18=42 op19=216 op20=172/>
// CHECK-1-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'f'
// CHECK-1-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-1-NEXT:     <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
// CHECK-1-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-1-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-1-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-1-NEXT:   </ReferenceBlock>
// CHECK-1-NEXT:   <DefLocation abbrevid=6 op0=17 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:   <Location abbrevid=7 op0=11 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-1-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-1-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-1-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-1-NEXT:     </ReferenceBlock>
// CHECK-1-NEXT:   </TypeBlock>
// CHECK-1-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/9A82CB33ED0FDF81EE383D31CD0957D153C5E840.bc | FileCheck %s --check-prefix CHECK-2
// CHECK-2: <BLOCKINFO_BLOCK/>
// CHECK-2-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-2-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-2-NEXT: </VersionBlock>
// CHECK-2-NEXT: <FunctionBlock NumWords=56 BlockCodeSize=4>
// CHECK-2-NEXT:   <USR abbrevid=4 op0=20 op1=154 op2=130 op3=203 op4=51 op5=237 op6=15 op7=223 op8=129 op9=238 op10=56 op11=61 op12=49 op13=205 op14=9 op15=87 op16=209 op17=83 op18=197 op19=232 op20=64/>
// CHECK-2-NEXT:   <Name abbrevid=5 op0=4/> blob data = 'func'
// CHECK-2-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-2-NEXT:     <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
// CHECK-2-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'B'
// CHECK-2-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-2-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-2-NEXT:   </ReferenceBlock>
// CHECK-2-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-2-NEXT:     <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
// CHECK-2-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-2-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-2-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-2-NEXT:   </ReferenceBlock>
// CHECK-2-NEXT:   <DefLocation abbrevid=6 op0=23 op1=4/> blob data = '{{.*}}'
// CHECK-2-NEXT:   <TypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-2-NEXT:     <ReferenceBlock NumWords=5 BlockCodeSize=4>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=12/> blob data = 'enum A::B::E'
// CHECK-2-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:   </TypeBlock>
// CHECK-2-NEXT:   <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-2-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-2-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'i'
// CHECK-2-NEXT:   </FieldTypeBlock>
// CHECK-2-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/8D042EFFC98B373450BC6B5B90A330C25A150E9C.bc | FileCheck %s --check-prefix CHECK-3
// CHECK-3: <BLOCKINFO_BLOCK/>
// CHECK-3-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-3-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-3-NEXT: </VersionBlock>
// CHECK-3-NEXT: <NamespaceBlock NumWords=9 BlockCodeSize=4>
// CHECK-3-NEXT:   <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
// CHECK-3-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-3-NEXT: </NamespaceBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/E21AF79E2A9D02554BA090D10DF39FE273F5CDB5.bc | FileCheck %s --check-prefix CHECK-4
// CHECK-4: <BLOCKINFO_BLOCK/>
// CHECK-4-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-4-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-4-NEXT: </VersionBlock>
// CHECK-4-NEXT: <NamespaceBlock NumWords=21 BlockCodeSize=4>
// CHECK-4-NEXT:   <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
// CHECK-4-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'B'
// CHECK-4-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-4-NEXT:     <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
// CHECK-4-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-4-NEXT:     <RefType abbrevid=6 op0=1/>
// CHECK-4-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-4-NEXT:   </ReferenceBlock>
// CHECK-4-NEXT: </NamespaceBlock>
