// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-intermediate -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/8D042EFFC98B373450BC6B5B90A330C25A150E9C.bc --dump | FileCheck %s --check-prefix CHECK-A
// RUN: llvm-bcanalyzer %t/docs/bc/E21AF79E2A9D02554BA090D10DF39FE273F5CDB5.bc --dump | FileCheck %s --check-prefix CHECK-B
// RUN: llvm-bcanalyzer %t/docs/bc/39D3C95A5F7CE2BA4937BD7B01BAE09EBC2AD8AC.bc --dump | FileCheck %s --check-prefix CHECK-F
// RUN: llvm-bcanalyzer %t/docs/bc/9A82CB33ED0FDF81EE383D31CD0957D153C5E840.bc --dump | FileCheck %s --check-prefix CHECK-FUNC
// RUN: llvm-bcanalyzer %t/docs/bc/E9ABF7E7E2425B626723D41E76E4BC7E7A5BD775.bc --dump | FileCheck %s --check-prefix CHECK-E
 
namespace A {
// CHECK-A: <NamespaceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-A-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
  // CHECK-A-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-A-NEXT: </NamespaceBlock>
  
void f();

}  // namespace A

namespace A {

void f(){};
// CHECK-F: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-F-NEXT: <USR abbrevid=4 op0=20 op1=57 op2=211 op3=201 op4=90 op5=95 op6=124 op7=226 op8=186 op9=73 op10=55 op11=189 op12=123 op13=1 op14=186 op15=224 op16=158 op17=188 op18=42 op19=216 op20=172/>
  // CHECK-F-NEXT: <Name abbrevid=5 op0=1/> blob data = 'f'
  // CHECK-F-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-F-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
    // CHECK-F-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
    // CHECK-F-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-F-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-F-NEXT: </ReferenceBlock>
  // CHECK-F-NEXT: <DefLocation abbrevid=6 op0=24 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-F-NEXT: <Location abbrevid=7 op0=18 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-F-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-F-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-F-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-F-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-F-NEXT: </ReferenceBlock>
  // CHECK-F-NEXT: </TypeBlock>
// CHECK-F-NEXT: </FunctionBlock>

namespace B {
// CHECK-B: <NamespaceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-B-NEXT: <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
  // CHECK-B-NEXT: <Name abbrevid=5 op0=1/> blob data = 'B'
  // CHECK-B-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-B-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
    // CHECK-B-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
    // CHECK-B-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-B-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-B-NEXT: </ReferenceBlock>
// CHECK-B-NEXT: </NamespaceBlock>

enum E { X };
// CHECK-E: <EnumBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-E-NEXT: <USR abbrevid=4 op0=20 op1=233 op2=171 op3=247 op4=231 op5=226 op6=66 op7=91 op8=98 op9=103 op10=35 op11=212 op12=30 op13=118 op14=228 op15=188 op16=126 op17=122 op18=91 op19=215 op20=117/>
  // CHECK-E-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
  // CHECK-E-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-E-NEXT: <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
    // CHECK-E-NEXT: <Name abbrevid=5 op0=1/> blob data = 'B'
    // CHECK-E-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-E-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-E-NEXT: </ReferenceBlock>
  // CHECK-E-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-E-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
    // CHECK-E-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
    // CHECK-E-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-E-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-E-NEXT: </ReferenceBlock>
  // CHECK-E-NEXT: <DefLocation abbrevid=6 op0=56 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-E-NEXT: <Member abbrevid=8 op0=1/> blob data = 'X'
// CHECK-E-NEXT: </EnumBlock>

E func(int i) { return X; }
// CHECK-FUNC: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-FUNC-NEXT: <USR abbrevid=4 op0=20 op1=154 op2=130 op3=203 op4=51 op5=237 op6=15 op7=223 op8=129 op9=238 op10=56 op11=61 op12=49 op13=205 op14=9 op15=87 op16=209 op17=83 op18=197 op19=232 op20=64/>
  // CHECK-FUNC-NEXT: <Name abbrevid=5 op0=4/> blob data = 'func'
  // CHECK-FUNC-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-FUNC-NEXT: <USR abbrevid=4 op0=20 op1=226 op2=26 op3=247 op4=158 op5=42 op6=157 op7=2 op8=85 op9=75 op10=160 op11=144 op12=209 op13=13 op14=243 op15=159 op16=226 op17=115 op18=245 op19=205 op20=181/>
    // CHECK-FUNC-NEXT: <Name abbrevid=5 op0=1/> blob data = 'B'
    // CHECK-FUNC-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-FUNC-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-FUNC-NEXT: </ReferenceBlock>
  // CHECK-FUNC-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-FUNC-NEXT: <USR abbrevid=4 op0=20 op1=141 op2=4 op3=46 op4=255 op5=201 op6=139 op7=55 op8=52 op9=80 op10=188 op11=107 op12=91 op13=144 op14=163 op15=48 op16=194 op17=90 op18=21 op19=14 op20=156/>
    // CHECK-FUNC-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
    // CHECK-FUNC-NEXT: <RefType abbrevid=6 op0=1/>
    // CHECK-FUNC-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-FUNC-NEXT: </ReferenceBlock>
  // CHECK-FUNC-NEXT: <DefLocation abbrevid=6 op0=76 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-FUNC-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-FUNC-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-FUNC-NEXT: <Name abbrevid=5 op0=12/> blob data = 'enum A::B::E'
      // CHECK-FUNC-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-FUNC-NEXT: </ReferenceBlock>
  // CHECK-FUNC-NEXT: </TypeBlock>
  // CHECK-FUNC-NEXT: <FieldTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-FUNC-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-FUNC-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-FUNC-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-FUNC-NEXT: </ReferenceBlock>
    // CHECK-FUNC-NEXT: <Name abbrevid=4 op0=1/> blob data = 'i'
  // CHECK-FUNC-NEXT: </FieldTypeBlock>
// CHECK-FUNC-NEXT: </FunctionBlock>

}  // namespace B
}  // namespace A
