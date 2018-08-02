// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// This test requires Linux due to system-dependent USR for the inner class.
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void H() {
  class I {};
}

union A { int X; int Y; };

enum B { X, Y };

enum class Bc { A, B };

struct C { int i; };

class D {};

class E {
public:
  E() {}
  ~E() {}

protected:
  void ProtectedMethod();
};

void E::ProtectedMethod() {}

class F : virtual private D, public E {};

class X {
  class Y {};
};

// RUN: clang-doc --dump-intermediate --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/289584A8E0FF4178A794622A547AA622503967A1.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-0-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-0-NEXT:   <DefLocation abbrevid=6 op0=25 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-0-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/3FB542274573CAEAD54CEBFFCAEE3D77FB9713D8.bc | FileCheck %s --check-prefix CHECK-1
// CHECK-1: <BLOCKINFO_BLOCK/>
// CHECK-1-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-1-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-1-NEXT: </VersionBlock>
// CHECK-1-NEXT: <RecordBlock NumWords=24 BlockCodeSize=4>
// CHECK-1-NEXT:   <USR abbrevid=4 op0=20 op1=63 op2=181 op3=66 op4=39 op5=69 op6=115 op7=202 op8=234 op9=213 op10=76 op11=235 op12=255 op13=202 op14=238 op15=61 op16=119 op17=251 op18=151 op19=19 op20=216/>
// CHECK-1-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'I'
// CHECK-1-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-1-NEXT:     <USR abbrevid=4 op0=20 op1=182 op2=172 op3=76 op4=92 op5=159 op6=46 op7=163 op8=242 op9=179 op10=236 op11=225 op12=163 op13=61 op14=52 op15=159 op16=78 op17=229 op18=2 op19=178 op20=78/>
// CHECK-1-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'H'
// CHECK-1-NEXT:     <RefType abbrevid=6 op0=3/>
// CHECK-1-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-1-NEXT:   </ReferenceBlock>
// CHECK-1-NEXT:   <DefLocation abbrevid=6 op0=12 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-1-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/5093D428CDC62096A67547BA52566E4FB9404EEE.bc | FileCheck %s --check-prefix CHECK-2
// CHECK-2: <BLOCKINFO_BLOCK/>
// CHECK-2-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-2-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-2-NEXT: </VersionBlock>
// CHECK-2-NEXT: <FunctionBlock NumWords=50 BlockCodeSize=4>
// CHECK-2-NEXT:   <USR abbrevid=4 op0=20 op1=80 op2=147 op3=212 op4=40 op5=205 op6=198 op7=32 op8=150 op9=166 op10=117 op11=71 op12=186 op13=82 op14=86 op15=110 op16=79 op17=185 op18=64 op19=78 op20=238/>
// CHECK-2-NEXT:   <Name abbrevid=5 op0=15/> blob data = 'ProtectedMethod'
// CHECK-2-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-2-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-2-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-2-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-2-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-2-NEXT:   </ReferenceBlock>
// CHECK-2-NEXT:   <IsMethod abbrevid=9 op0=1/>
// CHECK-2-NEXT:   <DefLocation abbrevid=6 op0=34 op1=4/> blob data = '{{.*}}'
// CHECK-2-NEXT:   <Location abbrevid=7 op0=31 op1=4/> blob data = '{{.*}}'
// CHECK-2-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-2-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-2-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-2-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-2-NEXT:     <Field abbrevid=7 op0=2/>
// CHECK-2-NEXT:   </ReferenceBlock>
// CHECK-2-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-2-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-2-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:   </TypeBlock>
// CHECK-2-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/CA7C7935730B5EACD25F080E9C83FA087CCDC75E.bc | FileCheck %s --check-prefix CHECK-3
// CHECK-3: <BLOCKINFO_BLOCK/>
// CHECK-3-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-3-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-3-NEXT: </VersionBlock>
// CHECK-3-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-3-NEXT:   <USR abbrevid=4 op0=20 op1=202 op2=124 op3=121 op4=53 op5=115 op6=11 op7=94 op8=172 op9=210 op10=95 op11=8 op12=14 op13=156 op14=131 op15=250 op16=8 op17=124 op18=205 op19=199 op20=94/>
// CHECK-3-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'X'
// CHECK-3-NEXT:   <DefLocation abbrevid=6 op0=38 op1=4/> blob data = '{{.*}}'
// CHECK-3-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-3-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E.bc | FileCheck %s --check-prefix CHECK-4
// CHECK-4: <BLOCKINFO_BLOCK/>
// CHECK-4-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-4-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-4-NEXT: </VersionBlock>
// CHECK-4-NEXT: <FunctionBlock NumWords=20 BlockCodeSize=4>
// CHECK-4-NEXT:   <USR abbrevid=4 op0=20 op1=182 op2=172 op3=76 op4=92 op5=159 op6=46 op7=163 op8=242 op9=179 op10=236 op11=225 op12=163 op13=61 op14=52 op15=159 op16=78 op17=229 op18=2 op19=178 op20=78/>
// CHECK-4-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'H'
// CHECK-4-NEXT:   <DefLocation abbrevid=6 op0=11 op1=4/> blob data = '{{.*}}'
// CHECK-4-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-4-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-4-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-4-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-4-NEXT:     </ReferenceBlock>
// CHECK-4-NEXT:   </TypeBlock>
// CHECK-4-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/06B5F6A19BA9F6A832E127C9968282B94619B210.bc | FileCheck %s --check-prefix CHECK-5
// CHECK-5: <BLOCKINFO_BLOCK/>
// CHECK-5-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-5-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-5-NEXT: </VersionBlock>
// CHECK-5-NEXT: <RecordBlock NumWords=22 BlockCodeSize=4>
// CHECK-5-NEXT:   <USR abbrevid=4 op0=20 op1=6 op2=181 op3=246 op4=161 op5=155 op6=169 op7=246 op8=168 op9=50 op10=225 op11=39 op12=201 op13=150 op14=130 op15=130 op16=185 op17=70 op18=25 op19=178 op20=16/>
// CHECK-5-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'C'
// CHECK-5-NEXT:   <DefLocation abbrevid=6 op0=21 op1=4/> blob data = '{{.*}}'
// CHECK-5-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-5-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-5-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-5-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-5-NEXT:     </ReferenceBlock>
// CHECK-5-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'i'
// CHECK-5-NEXT:     <Access abbrevid=5 op0=3/>
// CHECK-5-NEXT:   </MemberTypeBlock>
// CHECK-5-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/BD2BDEBD423F80BACCEA75DE6D6622D355FC2D17.bc | FileCheck %s --check-prefix CHECK-6
// CHECK-6: <BLOCKINFO_BLOCK/>
// CHECK-6-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-6-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-6-NEXT: </VersionBlock>
// CHECK-6-NEXT: <FunctionBlock NumWords=44 BlockCodeSize=4>
// CHECK-6-NEXT:   <USR abbrevid=4 op0=20 op1=189 op2=43 op3=222 op4=189 op5=66 op6=63 op7=128 op8=186 op9=204 op10=234 op11=117 op12=222 op13=109 op14=102 op15=34 op16=211 op17=85 op18=252 op19=45 op20=23/>
// CHECK-6-NEXT:   <Name abbrevid=5 op0=2/> blob data = '~E'
// CHECK-6-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-6-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-6-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-6-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-6-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-6-NEXT:   </ReferenceBlock>
// CHECK-6-NEXT:   <IsMethod abbrevid=9 op0=1/>
// CHECK-6-NEXT:   <DefLocation abbrevid=6 op0=28 op1=4/> blob data = '{{.*}}'
// CHECK-6-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-6-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-6-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-6-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-6-NEXT:     <Field abbrevid=7 op0=2/>
// CHECK-6-NEXT:   </ReferenceBlock>
// CHECK-6-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-6-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-6-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-6-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-6-NEXT:     </ReferenceBlock>
// CHECK-6-NEXT:   </TypeBlock>
// CHECK-6-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/DEB4AC1CD9253CD9EF7FBE6BCAC506D77984ABD4.bc | FileCheck %s --check-prefix CHECK-7
// CHECK-7: <BLOCKINFO_BLOCK/>
// CHECK-7-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-7-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-7-NEXT: </VersionBlock>
// CHECK-7-NEXT: <FunctionBlock NumWords=44 BlockCodeSize=4>
// CHECK-7-NEXT:   <USR abbrevid=4 op0=20 op1=222 op2=180 op3=172 op4=28 op5=217 op6=37 op7=60 op8=217 op9=239 op10=127 op11=190 op12=107 op13=202 op14=197 op15=6 op16=215 op17=121 op18=132 op19=171 op20=212/>
// CHECK-7-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-7-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-7-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-7-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-7-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-7-NEXT:   </ReferenceBlock>
// CHECK-7-NEXT:   <IsMethod abbrevid=9 op0=1/>
// CHECK-7-NEXT:   <DefLocation abbrevid=6 op0=27 op1=4/> blob data = '{{.*}}'
// CHECK-7-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-7-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-7-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-7-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:     <Field abbrevid=7 op0=2/>
// CHECK-7-NEXT:   </ReferenceBlock>
// CHECK-7-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-7-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-7-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-7-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-7-NEXT:     </ReferenceBlock>
// CHECK-7-NEXT:   </TypeBlock>
// CHECK-7-NEXT: </FunctionBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/641AB4A3D36399954ACDE29C7A8833032BF40472.bc | FileCheck %s --check-prefix CHECK-8
// CHECK-8: <BLOCKINFO_BLOCK/>
// CHECK-8-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-8-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-8-NEXT: </VersionBlock>
// CHECK-8-NEXT: <RecordBlock NumWords=24 BlockCodeSize=4>
// CHECK-8-NEXT:   <USR abbrevid=4 op0=20 op1=100 op2=26 op3=180 op4=163 op5=211 op6=99 op7=153 op8=149 op9=74 op10=205 op11=226 op12=156 op13=122 op14=136 op15=51 op16=3 op17=43 op18=244 op19=4 op20=114/>
// CHECK-8-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'Y'
// CHECK-8-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-8-NEXT:     <USR abbrevid=4 op0=20 op1=202 op2=124 op3=121 op4=53 op5=115 op6=11 op7=94 op8=172 op9=210 op10=95 op11=8 op12=14 op13=156 op14=131 op15=250 op16=8 op17=124 op18=205 op19=199 op20=94/>
// CHECK-8-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'X'
// CHECK-8-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-8-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-8-NEXT:   </ReferenceBlock>
// CHECK-8-NEXT:   <DefLocation abbrevid=6 op0=39 op1=4/> blob data = '{{.*}}'
// CHECK-8-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-8-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/FC07BD34D5E77782C263FA944447929EA8753740.bc | FileCheck %s --check-prefix CHECK-9
// CHECK-9: <BLOCKINFO_BLOCK/>
// CHECK-9-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-9-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-9-NEXT: </VersionBlock>
// CHECK-9-NEXT: <EnumBlock NumWords=16 BlockCodeSize=4>
// CHECK-9-NEXT:   <USR abbrevid=4 op0=20 op1=252 op2=7 op3=189 op4=52 op5=213 op6=231 op7=119 op8=130 op9=194 op10=99 op11=250 op12=148 op13=68 op14=71 op15=146 op16=158 op17=168 op18=117 op19=55 op20=64/>
// CHECK-9-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'B'
// CHECK-9-NEXT:   <DefLocation abbrevid=6 op0=17 op1=4/> blob data = '{{.*}}'
// CHECK-9-NEXT:   <Member abbrevid=8 op0=1/> blob data = 'X'
// CHECK-9-NEXT:   <Member abbrevid=8 op0=1/> blob data = 'Y'
// CHECK-9-NEXT: </EnumBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/0921737541208B8FA9BB42B60F78AC1D779AA054.bc | FileCheck %s --check-prefix CHECK-10
// CHECK-10: <BLOCKINFO_BLOCK/>
// CHECK-10-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-10-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-10-NEXT: </VersionBlock>
// CHECK-10-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-10-NEXT:   <USR abbrevid=4 op0=20 op1=9 op2=33 op3=115 op4=117 op5=65 op6=32 op7=139 op8=143 op9=169 op10=187 op11=66 op12=182 op13=15 op14=120 op15=172 op16=29 op17=119 op18=154 op19=160 op20=84/>
// CHECK-10-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'D'
// CHECK-10-NEXT:   <DefLocation abbrevid=6 op0=23 op1=4/> blob data = '{{.*}}'
// CHECK-10-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-10-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/E3B54702FABFF4037025BA194FC27C47006330B5.bc | FileCheck %s --check-prefix CHECK-11
// CHECK-11: <BLOCKINFO_BLOCK/>
// CHECK-11-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-11-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-11-NEXT: </VersionBlock>
// CHECK-11-NEXT: <RecordBlock NumWords=37 BlockCodeSize=4>
// CHECK-11-NEXT:   <USR abbrevid=4 op0=20 op1=227 op2=181 op3=71 op4=2 op5=250 op6=191 op7=244 op8=3 op9=112 op10=37 op11=186 op12=25 op13=79 op14=194 op15=124 op16=71 op17=0 op18=99 op19=48 op20=181/>
// CHECK-11-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'F'
// CHECK-11-NEXT:   <DefLocation abbrevid=6 op0=36 op1=4/> blob data = '{{.*}}'
// CHECK-11-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-11-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-11-NEXT:     <USR abbrevid=4 op0=20 op1=40 op2=149 op3=132 op4=168 op5=224 op6=255 op7=65 op8=120 op9=167 op10=148 op11=98 op12=42 op13=84 op14=122 op15=166 op16=34 op17=80 op18=57 op19=103 op20=161/>
// CHECK-11-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-11-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-11-NEXT:     <Field abbrevid=7 op0=2/>
// CHECK-11-NEXT:   </ReferenceBlock>
// CHECK-11-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-11-NEXT:     <USR abbrevid=4 op0=20 op1=9 op2=33 op3=115 op4=117 op5=65 op6=32 op7=139 op8=143 op9=169 op10=187 op11=66 op12=182 op13=15 op14=120 op15=172 op16=29 op17=119 op18=154 op19=160 op20=84/>
// CHECK-11-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'D'
// CHECK-11-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-11-NEXT:     <Field abbrevid=7 op0=3/>
// CHECK-11-NEXT:   </ReferenceBlock>
// CHECK-11-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/ACE81AFA6627B4CEF2B456FB6E1252925674AF7E.bc | FileCheck %s --check-prefix CHECK-12
// CHECK-12: <BLOCKINFO_BLOCK/>
// CHECK-12-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-12-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-12-NEXT: </VersionBlock>
// CHECK-12-NEXT: <RecordBlock NumWords=33 BlockCodeSize=4>
// CHECK-12-NEXT:   <USR abbrevid=4 op0=20 op1=172 op2=232 op3=26 op4=250 op5=102 op6=39 op7=180 op8=206 op9=242 op10=180 op11=86 op12=251 op13=110 op14=18 op15=82 op16=146 op17=86 op18=116 op19=175 op20=126/>
// CHECK-12-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-12-NEXT:   <DefLocation abbrevid=6 op0=15 op1=4/> blob data = '{{.*}}'
// CHECK-12-NEXT:   <TagType abbrevid=8 op0=2/>
// CHECK-12-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-12-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-12-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-12-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-12-NEXT:     </ReferenceBlock>
// CHECK-12-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'X'
// CHECK-12-NEXT:     <Access abbrevid=5 op0=3/>
// CHECK-12-NEXT:   </MemberTypeBlock>
// CHECK-12-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-12-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-12-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-12-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-12-NEXT:     </ReferenceBlock>
// CHECK-12-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'Y'
// CHECK-12-NEXT:     <Access abbrevid=5 op0=3/>
// CHECK-12-NEXT:   </MemberTypeBlock>
// CHECK-12-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/1E3438A08BA22025C0B46289FF0686F92C8924C5.bc | FileCheck %s --check-prefix CHECK-13
// CHECK-13: <BLOCKINFO_BLOCK/>
// CHECK-13-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-13-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-13-NEXT: </VersionBlock>
// CHECK-13-NEXT: <EnumBlock NumWords=16 BlockCodeSize=4>
// CHECK-13-NEXT:   <USR abbrevid=4 op0=20 op1=30 op2=52 op3=56 op4=160 op5=139 op6=162 op7=32 op8=37 op9=192 op10=180 op11=98 op12=137 op13=255 op14=6 op15=134 op16=249 op17=44 op18=137 op19=36 op20=197/>
// CHECK-13-NEXT:   <Name abbrevid=5 op0=2/> blob data = 'Bc'
// CHECK-13-NEXT:   <DefLocation abbrevid=6 op0=19 op1=4/> blob data = '{{.*}}'
// CHECK-13-NEXT:   <Scoped abbrevid=9 op0=1/>
// CHECK-13-NEXT:   <Member abbrevid=8 op0=1/> blob data = 'A'
// CHECK-13-NEXT:   <Member abbrevid=8 op0=1/> blob data = 'B'
// CHECK-13-NEXT: </EnumBlock>
