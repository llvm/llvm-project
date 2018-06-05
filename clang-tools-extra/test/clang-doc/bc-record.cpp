// This test requires Linux due to the system-dependent USR for the
// inner class in function H.
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-intermediate -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/ACE81AFA6627B4CEF2B456FB6E1252925674AF7E.bc --dump | FileCheck %s --check-prefix CHECK-A
// RUN: llvm-bcanalyzer %t/docs/bc/FC07BD34D5E77782C263FA944447929EA8753740.bc --dump | FileCheck %s --check-prefix CHECK-B
// RUN: llvm-bcanalyzer %t/docs/bc/1E3438A08BA22025C0B46289FF0686F92C8924C5.bc --dump | FileCheck %s --check-prefix CHECK-BC
// RUN: llvm-bcanalyzer %t/docs/bc/06B5F6A19BA9F6A832E127C9968282B94619B210.bc --dump | FileCheck %s --check-prefix CHECK-C
// RUN: llvm-bcanalyzer %t/docs/bc/0921737541208B8FA9BB42B60F78AC1D779AA054.bc --dump | FileCheck %s --check-prefix CHECK-D
// RUN: llvm-bcanalyzer %t/docs/bc/289584A8E0FF4178A794622A547AA622503967A1.bc --dump | FileCheck %s --check-prefix CHECK-E
// RUN: llvm-bcanalyzer %t/docs/bc/DEB4AC1CD9253CD9EF7FBE6BCAC506D77984ABD4.bc --dump | FileCheck %s --check-prefix CHECK-ECON
// RUN: llvm-bcanalyzer %t/docs/bc/BD2BDEBD423F80BACCEA75DE6D6622D355FC2D17.bc --dump | FileCheck %s --check-prefix CHECK-EDES
// RUN: llvm-bcanalyzer %t/docs/bc/E3B54702FABFF4037025BA194FC27C47006330B5.bc --dump | FileCheck %s --check-prefix CHECK-F
// RUN: llvm-bcanalyzer %t/docs/bc/B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E.bc --dump | FileCheck %s --check-prefix CHECK-H
// RUN: llvm-bcanalyzer %t/docs/bc/6BA1EE2B3DAEACF6E4306F10AF44908F4807927C.bc --dump | FileCheck %s --check-prefix CHECK-I
// RUN: llvm-bcanalyzer %t/docs/bc/5093D428CDC62096A67547BA52566E4FB9404EEE.bc --dump | FileCheck %s --check-prefix CHECK-PM
// RUN: llvm-bcanalyzer %t/docs/bc/CA7C7935730B5EACD25F080E9C83FA087CCDC75E.bc --dump | FileCheck %s --check-prefix CHECK-X
// RUN: llvm-bcanalyzer %t/docs/bc/641AB4A3D36399954ACDE29C7A8833032BF40472.bc --dump | FileCheck %s --check-prefix CHECK-Y

void H() {
  class I {};
}
// CHECK-H: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-H-NEXT: <USR abbrevid=4 op0=20 op1=182 op2=172 op3=76 op4=92 op5=159 op6=46 op7=163 op8=242 op9=179 op10=236 op11=225 op12=163 op13=61 op14=52 op15=159 op16=78 op17=229 op18=2 op19=178 op20=78/>
  // CHECK-H-NEXT: <Name abbrevid=5 op0=1/> blob data = 'H'
  // CHECK-H-NEXT: <DefLocation abbrevid=6 op0=24 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-H-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-H-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-H-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-H-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-H-NEXT: </ReferenceBlock>
  // CHECK-H-NEXT: </TypeBlock>
// CHECK-H-NEXT: </FunctionBlock>


// CHECK-I: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-I-NEXT: <USR abbrevid=4 op0=20 op1=107 op2=161 op3=238 op4=43 op5=61 op6=174 op7=172 op8=246 op9=228 op10=48 op11=111 op12=16 op13=175 op14=68 op15=144 op16=143 op17=72 op18=7 op19=146 op20=124/>
  // CHECK-I-NEXT: <Name abbrevid=5 op0=1/> blob data = 'I'
  // CHECK-I-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-I-NEXT: <USR abbrevid=4 op0=20 op1=182 op2=172 op3=76 op4=92 op5=159 op6=46 op7=163 op8=242 op9=179 op10=236 op11=225 op12=163 op13=61 op14=52 op15=159 op16=78 op17=229 op18=2 op19=178 op20=78/>
    // CHECK-I-NEXT: <Name abbrevid=5 op0=1/> blob data = 'H'
    // CHECK-I-NEXT: <RefType abbrevid=6 op0=3/>
    // CHECK-I-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-I-NEXT: </ReferenceBlock>
  // CHECK-I-NEXT: <DefLocation abbrevid=6 op0=25 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-I-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-I-NEXT: </RecordBlock>

union A { int X; int Y; };
// CHECK-A: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-A-NEXT: <USR abbrevid=4 op0=20 op1=172 op2=232 op3=26 op4=250 op5=102 op6=39 op7=180 op8=206 op9=242 op10=180 op11=86 op12=251 op13=110 op14=18 op15=82 op16=146 op17=86 op18=116 op19=175 op20=126/>
  // CHECK-A-NEXT: <Name abbrevid=5 op0=1/> blob data = 'A'
  // CHECK-A-NEXT: <DefLocation abbrevid=6 op0=53 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-A-NEXT: <TagType abbrevid=8 op0=2/>
  // CHECK-A-NEXT: <MemberTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-A-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-A-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-A-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-A-NEXT: </ReferenceBlock>
    // CHECK-A-NEXT: <Name abbrevid=4 op0=1/> blob data = 'X'
    // CHECK-A-NEXT: <Access abbrevid=5 op0=3/>
  // CHECK-A-NEXT: </MemberTypeBlock>
  // CHECK-A-NEXT: <MemberTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-A-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-A-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-A-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-A-NEXT: </ReferenceBlock>
    // CHECK-A-NEXT: <Name abbrevid=4 op0=1/> blob data = 'Y'
    // CHECK-A-NEXT: <Access abbrevid=5 op0=3/>
  // CHECK-A-NEXT: </MemberTypeBlock>
// CHECK-A-NEXT: </RecordBlock>

enum B { X, Y };
// CHECK-B: <EnumBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-B-NEXT: <USR abbrevid=4 op0=20 op1=252 op2=7 op3=189 op4=52 op5=213 op6=231 op7=119 op8=130 op9=194 op10=99 op11=250 op12=148 op13=68 op14=71 op15=146 op16=158 op17=168 op18=117 op19=55 op20=64/>
  // CHECK-B-NEXT: <Name abbrevid=5 op0=1/> blob data = 'B'
  // CHECK-B-NEXT: <DefLocation abbrevid=6 op0=77 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-B-NEXT: <Member abbrevid=8 op0=1/> blob data = 'X'
  // CHECK-B-NEXT: <Member abbrevid=8 op0=1/> blob data = 'Y'
// CHECK-B-NEXT: </EnumBlock>

enum class Bc { A, B };
// CHECK-BC: <EnumBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-BC-NEXT: <USR abbrevid=4 op0=20 op1=30 op2=52 op3=56 op4=160 op5=139 op6=162 op7=32 op8=37 op9=192 op10=180 op11=98 op12=137 op13=255 op14=6 op15=134 op16=249 op17=44 op18=137 op19=36 op20=197/>
  // CHECK-BC-NEXT: <Name abbrevid=5 op0=2/> blob data = 'Bc'
  // CHECK-BC-NEXT: <DefLocation abbrevid=6 op0=86 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-BC-NEXT: <Scoped abbrevid=9 op0=1/>
  // CHECK-BC-NEXT: <Member abbrevid=8 op0=1/> blob data = 'A'
  // CHECK-BC-NEXT: <Member abbrevid=8 op0=1/> blob data = 'B'
// CHECK-BC-NEXT: </EnumBlock>

struct C { int i; };
// CHECK-C: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-C-NEXT: <USR abbrevid=4 op0=20 op1=6 op2=181 op3=246 op4=161 op5=155 op6=169 op7=246 op8=168 op9=50 op10=225 op11=39 op12=201 op13=150 op14=130 op15=130 op16=185 op17=70 op18=25 op19=178 op20=16/>
  // CHECK-C-NEXT: <Name abbrevid=5 op0=1/> blob data = 'C'
  // CHECK-C-NEXT: <DefLocation abbrevid=6 op0=96 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-C-NEXT: <MemberTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-C-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-C-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-C-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-C-NEXT: </ReferenceBlock>
    // CHECK-C-NEXT: <Name abbrevid=4 op0=1/> blob data = 'i'
    // CHECK-C-NEXT: <Access abbrevid=5 op0=3/>
  // CHECK-C-NEXT: </MemberTypeBlock>
// CHECK-C-NEXT: </RecordBlock>

class D {};
// CHECK-D: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-D-NEXT: <USR abbrevid=4 op0=20 op1=9 op2=33 op3=115 op4=117 op5=65 op6=32 op7=139 op8=143 op9=169 op10=187 op11=66 op12=182 op13=15 op14=120 op15=172 op16=29 op17=119 op18=154 op19=160 op20=84/>
  // CHECK-D-NEXT: <Name abbrevid=5 op0=1/> blob data = 'D'
  // CHECK-D-NEXT: <DefLocation abbrevid=6 op0=111 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-D-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-D-NEXT: </RecordBlock>

class E {
public:
  E() {}
  ~E() {}

protected:
  void ProtectedMethod();
};
// CHECK-E: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-E-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
  // CHECK-E-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
  // CHECK-E-NEXT: <DefLocation abbrevid=6 op0=119 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-E-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-E-NEXT: </RecordBlock>

// CHECK-ECON: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-ECON-NEXT: <USR abbrevid=4 op0=20 op1=222 op2=180 op3=172 op4=28 op5=217 op6=37 op7=60 op8=217 op9=239 op10=127 op11=190 op12=107 op13=202 op14=197 op15=6 op16=215 op17=121 op18=132 op19=171 op20=212/>
  // CHECK-ECON-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-ECON-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-ECON-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
      // CHECK-ECON-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
      // CHECK-ECON-NEXT: <RefType abbrevid=6 op0=2/>
      // CHECK-ECON-NEXT: <Field abbrevid=7 op0=1/>
    // CHECK-ECON-NEXT: </ReferenceBlock>
  // CHECK-ECON-NEXT: <IsMethod abbrevid=9 op0=1/>
  // CHECK-ECON-NEXT: <DefLocation abbrevid=6 op0=121 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-ECON-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-ECON-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-ECON-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-ECON-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-ECON-NEXT: <Field abbrevid=7 op0=2/>
  // CHECK-ECON-NEXT: </ReferenceBlock>
  // CHECK-ECON-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-ECON-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-ECON-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-ECON-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-ECON-NEXT: </ReferenceBlock>
  // CHECK-ECON-NEXT: </TypeBlock>
// CHECK-ECON-NEXT: </FunctionBlock>

// CHECK-EDES: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-EDES-NEXT: <USR abbrevid=4 op0=20 op1=189 op2=43 op3=222 op4=189 op5=66 op6=63 op7=128 op8=186 op9=204 op10=234 op11=117 op12=222 op13=109 op14=102 op15=34 op16=211 op17=85 op18=252 op19=45 op20=23/>
  // CHECK-EDES-NEXT: <Name abbrevid=5 op0=2/> blob data = '~E'
  // CHECK-EDES-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-EDES-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-EDES-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-EDES-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-EDES-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-EDES-NEXT: </ReferenceBlock>
  // CHECK-EDES-NEXT: <IsMethod abbrevid=9 op0=1/>
  // CHECK-EDES-NEXT: <DefLocation abbrevid=6 op0=122 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-EDES-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-EDES-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-EDES-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-EDES-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-EDES-NEXT: <Field abbrevid=7 op0=2/>
  // CHECK-EDES-NEXT: </ReferenceBlock>
  // CHECK-EDES-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-EDES-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-EDES-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-EDES-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-EDES-NEXT: </ReferenceBlock>
  // CHECK-EDES-NEXT: </TypeBlock>
// CHECK-EDES-NEXT: </FunctionBlock>

void E::ProtectedMethod() {}
// CHECK-PM: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-PM-NEXT: <USR abbrevid=4 op0=20 op1=80 op2=147 op3=212 op4=40 op5=205 op6=198 op7=32 op8=150 op9=166 op10=117 op11=71 op12=186 op13=82 op14=86 op15=110 op16=79 op17=185 op18=64 op19=78 op20=238/>
  // CHECK-PM-NEXT: <Name abbrevid=5 op0=15/> blob data = 'ProtectedMethod'
  // CHECK-PM-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-PM-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-PM-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-PM-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-PM-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-PM-NEXT: </ReferenceBlock>
  // CHECK-PM-NEXT: <IsMethod abbrevid=9 op0=1/>
  // CHECK-PM-NEXT: <DefLocation abbrevid=6 op0=184 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-PM-NEXT: <Location abbrevid=7 op0=125 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-PM-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-PM-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-PM-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-PM-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-PM-NEXT: <Field abbrevid=7 op0=2/>
  // CHECK-PM-NEXT: </ReferenceBlock>
  // CHECK-PM-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-PM-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-PM-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-PM-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-PM-NEXT: </ReferenceBlock>
  // CHECK-PM-NEXT: </TypeBlock>
// CHECK-PM-NEXT: </FunctionBlock>



class F : virtual private D, public E {};
// CHECK-F: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-F-NEXT: <USR abbrevid=4 op0=20 op1=227 op2=181 op3=71 op4=2 op5=250 op6=191 op7=244 op8=3 op9=112 op10=37 op11=186 op12=25 op13=79 op14=194 op15=124 op16=71 op17=0 op18=99 op19=48 op20=181/>
  // CHECK-F-NEXT: <Name abbrevid=5 op0=1/> blob data = 'F'
  // CHECK-F-NEXT: <DefLocation abbrevid=6 op0=213 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-F-NEXT: <TagType abbrevid=8 op0=3/>
  // CHECK-F-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-F-NEXT: <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
    // CHECK-F-NEXT: <Name abbrevid=5 op0=1/> blob data = 'E'
    // CHECK-F-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-F-NEXT: <Field abbrevid=7 op0=2/>
  // CHECK-F-NEXT: </ReferenceBlock>
  // CHECK-F-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-F-NEXT: <USR abbrevid=4 op0=20 op1=9 op2=33 op3=115 op4=117 op5=65 op6=32 op7=139 op8=143 op9=169 op10=187 op11=66 op12=182 op13=15 op14=120 op15=172 op16=29 op17=119 op18=154 op19=160 op20=84/>
    // CHECK-F-NEXT: <Name abbrevid=5 op0=1/> blob data = 'D'
    // CHECK-F-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-F-NEXT: <Field abbrevid=7 op0=3/>
  // CHECK-F-NEXT: </ReferenceBlock>
// CHECK-F-NEXT: </RecordBlock>

class X {
  class Y {};
};
// CHECK-X: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-X-NEXT: <USR abbrevid=4 op0=20 op1=202 op2=124 op3=121 op4=53 op5=115 op6=11 op7=94 op8=172 op9=210 op10=95 op11=8 op12=14 op13=156 op14=131 op15=250 op16=8 op17=124 op18=205 op19=199 op20=94/>
  // CHECK-X-NEXT: <Name abbrevid=5 op0=1/> blob data = 'X'
  // CHECK-X-NEXT: <DefLocation abbrevid=6 op0=233 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-X-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-X-NEXT: </RecordBlock>

// CHECK-Y: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-Y-NEXT: <USR abbrevid=4 op0=20 op1=100 op2=26 op3=180 op4=163 op5=211 op6=99 op7=153 op8=149 op9=74 op10=205 op11=226 op12=156 op13=122 op14=136 op15=51 op16=3 op17=43 op18=244 op19=4 op20=114/>
  // CHECK-Y-NEXT: <Name abbrevid=5 op0=1/> blob data = 'Y'
  // CHECK-Y-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-Y-NEXT: <USR abbrevid=4 op0=20 op1=202 op2=124 op3=121 op4=53 op5=115 op6=11 op7=94 op8=172 op9=210 op10=95 op11=8 op12=14 op13=156 op14=131 op15=250 op16=8 op17=124 op18=205 op19=199 op20=94/>
    // CHECK-Y-NEXT: <Name abbrevid=5 op0=1/> blob data = 'X'
    // CHECK-Y-NEXT: <RefType abbrevid=6 op0=2/>
    // CHECK-Y-NEXT: <Field abbrevid=7 op0=1/>
  // CHECK-Y-NEXT: </ReferenceBlock>
  // CHECK-Y-NEXT: <DefLocation abbrevid=6 op0=234 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-Y-NEXT: <TagType abbrevid=8 op0=3/>
// CHECK-Y-NEXT: </RecordBlock>
