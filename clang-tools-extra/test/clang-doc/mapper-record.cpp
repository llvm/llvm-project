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

class G;

// RUN: clang-doc --dump-mapper --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/289584A8E0FF4178A794622A547AA622503967A1.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <RecordBlock NumWords=55 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:   <FunctionBlock NumWords=47 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=15/> blob data = 'ProtectedMethod'
// CHECK-0-NEXT:     <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-0-NEXT:     <DefLocation abbrevid=6 op0=34 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:     </TypeBlock>
// CHECK-0-NEXT:   </FunctionBlock>
// CHECK-0-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/CA7C7935730B5EACD25F080E9C83FA087CCDC75E.bc | FileCheck %s --check-prefix CHECK-1
// CHECK-1: <BLOCKINFO_BLOCK/>
// CHECK-1-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-1-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-1-NEXT: </VersionBlock>
// CHECK-1-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-1-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'X'
// CHECK-1-NEXT:   <DefLocation abbrevid=6 op0=38 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-1-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/06B5F6A19BA9F6A832E127C9968282B94619B210.bc | FileCheck %s --check-prefix CHECK-2
// CHECK-2: <BLOCKINFO_BLOCK/>
// CHECK-2-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-2-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-2-NEXT: </VersionBlock>
// CHECK-2-NEXT: <RecordBlock NumWords=22 BlockCodeSize=4>
// CHECK-2-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'C'
// CHECK-2-NEXT:   <DefLocation abbrevid=6 op0=21 op1=4/> blob data = '{{.*}}'
// CHECK-2-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-2-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-2-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'i'
// CHECK-2-NEXT:   </MemberTypeBlock>
// CHECK-2-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/4202E8BF0ECB12AE354C8499C52725B0EE30AED5.bc | FileCheck %s --check-prefix CHECK-3
// CHECK-3: <BLOCKINFO_BLOCK/>
// CHECK-3-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-3-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-3-NEXT: </VersionBlock>
// CHECK-3-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-3-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-3-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'G'
// CHECK-3-NEXT:   <Location abbrevid=7 op0=42 op1=4/> blob data = '{{.*}}'
// CHECK-3-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-3-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/641AB4A3D36399954ACDE29C7A8833032BF40472.bc | FileCheck %s --check-prefix CHECK-4
// CHECK-4: <BLOCKINFO_BLOCK/>
// CHECK-4-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-4-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-4-NEXT: </VersionBlock>
// CHECK-4-NEXT: <RecordBlock NumWords=24 BlockCodeSize=4>
// CHECK-4-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'Y'
// CHECK-4-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-4-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'X'
// CHECK-4-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-4-NEXT:     <Field abbrevid=7 op0=1/>
// CHECK-4-NEXT:   </ReferenceBlock>
// CHECK-4-NEXT:   <DefLocation abbrevid=6 op0=39 op1=4/> blob data = '{{.*}}'
// CHECK-4-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-4-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/0000000000000000000000000000000000000000.bc | FileCheck %s --check-prefix CHECK-5
// CHECK-5: <BLOCKINFO_BLOCK/>
// CHECK-5-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-5-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-5-NEXT: </VersionBlock>
// CHECK-5-NEXT: <NamespaceBlock NumWords=19 BlockCodeSize=4>
// CHECK-5-NEXT:   <EnumBlock NumWords=16 BlockCodeSize=4>
// CHECK-5-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-5-NEXT:     <Name abbrevid=5 op0=2/> blob data = 'Bc'
// CHECK-5-NEXT:     <DefLocation abbrevid=6 op0=19 op1=4/> blob data = '{{.*}}'
// CHECK-5-NEXT:     <Scoped abbrevid=9 op0=1/>
// CHECK-5-NEXT:     <Member abbrevid=8 op0=1/> blob data = 'A'
// CHECK-5-NEXT:     <Member abbrevid=8 op0=1/> blob data = 'B'
// CHECK-5-NEXT:   </EnumBlock>
// CHECK-5-NEXT: </NamespaceBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/0921737541208B8FA9BB42B60F78AC1D779AA054.bc | FileCheck %s --check-prefix CHECK-6
// CHECK-6: <BLOCKINFO_BLOCK/>
// CHECK-6-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-6-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-6-NEXT: </VersionBlock>
// CHECK-6-NEXT: <RecordBlock NumWords=12 BlockCodeSize=4>
// CHECK-6-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'D'
// CHECK-6-NEXT:   <DefLocation abbrevid=6 op0=23 op1=4/> blob data = '{{.*}}'
// CHECK-6-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-6-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/E3B54702FABFF4037025BA194FC27C47006330B5.bc | FileCheck %s --check-prefix CHECK-7
// CHECK-7: <BLOCKINFO_BLOCK/>
// CHECK-7-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-7-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-7-NEXT: </VersionBlock>
// CHECK-7-NEXT: <RecordBlock NumWords=37 BlockCodeSize=4>
// CHECK-7-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'F'
// CHECK-7-NEXT:   <DefLocation abbrevid=6 op0=36 op1=4/> blob data = '{{.*}}'
// CHECK-7-NEXT:   <TagType abbrevid=8 op0=3/>
// CHECK-7-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-7-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'E'
// CHECK-7-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:     <Field abbrevid=7 op0=2/>
// CHECK-7-NEXT:   </ReferenceBlock>
// CHECK-7-NEXT:   <ReferenceBlock NumWords=10 BlockCodeSize=4>
// CHECK-7-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'D'
// CHECK-7-NEXT:     <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:     <Field abbrevid=7 op0=3/>
// CHECK-7-NEXT:   </ReferenceBlock>
// CHECK-7-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/ACE81AFA6627B4CEF2B456FB6E1252925674AF7E.bc | FileCheck %s --check-prefix CHECK-8
// CHECK-8: <BLOCKINFO_BLOCK/>
// CHECK-8-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-8-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-8-NEXT: </VersionBlock>
// CHECK-8-NEXT: <RecordBlock NumWords=33 BlockCodeSize=4>
// CHECK-8-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-8-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'A'
// CHECK-8-NEXT:   <DefLocation abbrevid=6 op0=15 op1=4/> blob data = '{{.*}}'
// CHECK-8-NEXT:   <TagType abbrevid=8 op0=2/>
// CHECK-8-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-8-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-8-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-8-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-8-NEXT:     </ReferenceBlock>
// CHECK-8-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'X'
// CHECK-8-NEXT:   </MemberTypeBlock>
// CHECK-8-NEXT:   <MemberTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-8-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-8-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-8-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-8-NEXT:     </ReferenceBlock>
// CHECK-8-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'Y'
// CHECK-8-NEXT:   </MemberTypeBlock>
// CHECK-8-NEXT: </RecordBlock>
