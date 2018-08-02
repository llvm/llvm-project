// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void function(int x);

inline int inlinedFunction(int x);

int functionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

inline int inlinedFunctionWithInnerClass(int x) {
  class InnerClass { //VisibleNoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

class Class {
public:
  void publicMethod();
  int publicField;

protected:
  void protectedMethod();
  int protectedField;

private:
  void privateMethod();
  int privateField;
};

namespace named {
class NamedClass {
public:
  void namedPublicMethod();
  int namedPublicField;

protected:
  void namedProtectedMethod();
  int namedProtectedField;

private:
  void namedPrivateMethod();
  int namedPrivateField;
};

void namedFunction();
static void namedStaticFunction();
inline void namedInlineFunction();
} // namespace named

static void staticFunction(int x); //Internal Linkage

static int staticFunctionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

namespace {
class AnonClass {
public:
  void anonPublicMethod();
  int anonPublicField;

protected:
  void anonProtectedMethod();
  int anonProtectedField;

private:
  void anonPrivateMethod();
  int anonPrivateField;
};

void anonFunction();
static void anonStaticFunction();
inline void anonInlineFunction();
} // namespace

// RUN: clang-doc --dump-mapper --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/C9B3B71ACDD84C5BB320D34E97677715CDB3EA32.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <RecordBlock NumWords=79 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:   <FunctionBlock NumWords=71 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=17/> blob data = 'innerPublicMethod'
// CHECK-0-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <ReferenceBlock NumWords=17 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=29/> blob data = 'inlinedFunctionWithInnerClass'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=3/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-0-NEXT:     <DefLocation abbrevid=6 op0=26 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-0-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-0-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-0-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:     </TypeBlock>
// CHECK-0-NEXT:   </FunctionBlock>
// CHECK-0-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/8960B5C9247D6F5C532756E53A1AD1240FA2146F.bc | FileCheck %s --check-prefix CHECK-1
// CHECK-1: <BLOCKINFO_BLOCK/>
// CHECK-1-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-1-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-1-NEXT: </VersionBlock>
// CHECK-1-NEXT: <NamespaceBlock NumWords=45 BlockCodeSize=4>
// CHECK-1-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:   <FunctionBlock NumWords=37 BlockCodeSize=4>
// CHECK-1-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:     <Name abbrevid=5 op0=19/> blob data = 'namedInlineFunction'
// CHECK-1-NEXT:     <ReferenceBlock NumWords=11 BlockCodeSize=4>
// CHECK-1-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-1-NEXT:       <Name abbrevid=5 op0=5/> blob data = 'named'
// CHECK-1-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-1-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-1-NEXT:     </ReferenceBlock>
// CHECK-1-NEXT:     <Location abbrevid=7 op0=63 op1=4/> blob data = '{{.*}}'
// CHECK-1-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-1-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-1-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-1-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-1-NEXT:       </ReferenceBlock>
// CHECK-1-NEXT:     </TypeBlock>
// CHECK-1-NEXT:   </FunctionBlock>
// CHECK-1-NEXT: </NamespaceBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/74A031CBE68C101F3E83F60ED17F20C11EC19D48.bc | FileCheck %s --check-prefix CHECK-2
// CHECK-2: <BLOCKINFO_BLOCK/>
// CHECK-2-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-2-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-2-NEXT: </VersionBlock>
// CHECK-2-NEXT: <RecordBlock NumWords=78 BlockCodeSize=4>
// CHECK-2-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:   <FunctionBlock NumWords=70 BlockCodeSize=4>
// CHECK-2-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:     <Name abbrevid=5 op0=17/> blob data = 'innerPublicMethod'
// CHECK-2-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-2-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-2-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-2-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:     <ReferenceBlock NumWords=16 BlockCodeSize=4>
// CHECK-2-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=28/> blob data = 'staticFunctionWithInnerClass'
// CHECK-2-NEXT:       <RefType abbrevid=6 op0=3/>
// CHECK-2-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-2-NEXT:     <DefLocation abbrevid=6 op0=71 op1=4/> blob data = '{{.*}}'
// CHECK-2-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-2-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-2-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-2-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-2-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-2-NEXT:     </ReferenceBlock>
// CHECK-2-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-2-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-2-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-2-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-2-NEXT:       </ReferenceBlock>
// CHECK-2-NEXT:     </TypeBlock>
// CHECK-2-NEXT:   </FunctionBlock>
// CHECK-2-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/7CDD73DCD6CD72F7E5CE25502810A182C66C4B45.bc | FileCheck %s --check-prefix CHECK-3
// CHECK-3: <BLOCKINFO_BLOCK/>
// CHECK-3-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-3-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-3-NEXT: </VersionBlock>
// CHECK-3-NEXT: <RecordBlock NumWords=57 BlockCodeSize=4>
// CHECK-3-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-3-NEXT:   <FunctionBlock NumWords=49 BlockCodeSize=4>
// CHECK-3-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-3-NEXT:     <Name abbrevid=5 op0=13/> blob data = 'privateMethod'
// CHECK-3-NEXT:     <ReferenceBlock NumWords=11 BlockCodeSize=4>
// CHECK-3-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-3-NEXT:       <Name abbrevid=5 op0=5/> blob data = 'Class'
// CHECK-3-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-3-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-3-NEXT:     </ReferenceBlock>
// CHECK-3-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-3-NEXT:     <Location abbrevid=7 op0=42 op1=4/> blob data = '{{.*}}'
// CHECK-3-NEXT:     <ReferenceBlock NumWords=11 BlockCodeSize=4>
// CHECK-3-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-3-NEXT:       <Name abbrevid=5 op0=5/> blob data = 'Class'
// CHECK-3-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-3-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-3-NEXT:     </ReferenceBlock>
// CHECK-3-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-3-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-3-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-3-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-3-NEXT:       </ReferenceBlock>
// CHECK-3-NEXT:     </TypeBlock>
// CHECK-3-NEXT:   </FunctionBlock>
// CHECK-3-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/85427901413EC77C961019EBB3ADEF7B0BAAFE78.bc | FileCheck %s --check-prefix CHECK-4
// CHECK-4: <BLOCKINFO_BLOCK/>
// CHECK-4-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-4-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-4-NEXT: </VersionBlock>
// CHECK-4-NEXT: <RecordBlock NumWords=77 BlockCodeSize=4>
// CHECK-4-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:   <FunctionBlock NumWords=69 BlockCodeSize=4>
// CHECK-4-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:     <Name abbrevid=5 op0=17/> blob data = 'innerPublicMethod'
// CHECK-4-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-4-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-4-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-4-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-4-NEXT:     </ReferenceBlock>
// CHECK-4-NEXT:     <ReferenceBlock NumWords=15 BlockCodeSize=4>
// CHECK-4-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:       <Name abbrevid=5 op0=22/> blob data = 'functionWithInnerClass'
// CHECK-4-NEXT:       <RefType abbrevid=6 op0=3/>
// CHECK-4-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-4-NEXT:     </ReferenceBlock>
// CHECK-4-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-4-NEXT:     <DefLocation abbrevid=6 op0=17 op1=4/> blob data = '{{.*}}'
// CHECK-4-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-4-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-4-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'InnerClass'
// CHECK-4-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-4-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-4-NEXT:     </ReferenceBlock>
// CHECK-4-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-4-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-4-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-4-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-4-NEXT:       </ReferenceBlock>
// CHECK-4-NEXT:     </TypeBlock>
// CHECK-4-NEXT:   </FunctionBlock>
// CHECK-4-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/0000000000000000000000000000000000000000.bc | FileCheck %s --check-prefix CHECK-5
// CHECK-5: <BLOCKINFO_BLOCK/>
// CHECK-5-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-5-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-5-NEXT: </VersionBlock>
// CHECK-5-NEXT: <NamespaceBlock NumWords=39 BlockCodeSize=4>
// CHECK-5-NEXT:   <FunctionBlock NumWords=36 BlockCodeSize=4>
// CHECK-5-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-5-NEXT:     <Name abbrevid=5 op0=28/> blob data = 'staticFunctionWithInnerClass'
// CHECK-5-NEXT:     <DefLocation abbrevid=6 op0=68 op1=4/> blob data = '{{.*}}'
// CHECK-5-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-5-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-5-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-5-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-5-NEXT:       </ReferenceBlock>
// CHECK-5-NEXT:     </TypeBlock>
// CHECK-5-NEXT:     <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-5-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-5-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-5-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-5-NEXT:       </ReferenceBlock>
// CHECK-5-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'x'
// CHECK-5-NEXT:     </FieldTypeBlock>
// CHECK-5-NEXT:   </FunctionBlock>
// CHECK-5-NEXT: </NamespaceBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/4712C5FA37B298A25501D1033C619B65B0ECC449.bc | FileCheck %s --check-prefix CHECK-6
// CHECK-6: <BLOCKINFO_BLOCK/>
// CHECK-6-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-6-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-6-NEXT: </VersionBlock>
// CHECK-6-NEXT: <RecordBlock NumWords=73 BlockCodeSize=4>
// CHECK-6-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:   <FunctionBlock NumWords=65 BlockCodeSize=4>
// CHECK-6-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:     <Name abbrevid=5 op0=18/> blob data = 'namedPrivateMethod'
// CHECK-6-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-6-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'NamedClass'
// CHECK-6-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-6-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-6-NEXT:     </ReferenceBlock>
// CHECK-6-NEXT:     <ReferenceBlock NumWords=11 BlockCodeSize=4>
// CHECK-6-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:       <Name abbrevid=5 op0=5/> blob data = 'named'
// CHECK-6-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-6-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-6-NEXT:     </ReferenceBlock>
// CHECK-6-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-6-NEXT:     <Location abbrevid=7 op0=57 op1=4/> blob data = '{{.*}}'
// CHECK-6-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-6-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-6-NEXT:       <Name abbrevid=5 op0=10/> blob data = 'NamedClass'
// CHECK-6-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-6-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-6-NEXT:     </ReferenceBlock>
// CHECK-6-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-6-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-6-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-6-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-6-NEXT:       </ReferenceBlock>
// CHECK-6-NEXT:     </TypeBlock>
// CHECK-6-NEXT:   </FunctionBlock>
// CHECK-6-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/6E8FB72A89761E77020BFCEE9A9A6E64B15CC2A9.bc | FileCheck %s --check-prefix CHECK-7
// CHECK-7: <BLOCKINFO_BLOCK/>
// CHECK-7-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-7-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-7-NEXT: </VersionBlock>
// CHECK-7-NEXT: <RecordBlock NumWords=69 BlockCodeSize=4>
// CHECK-7-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:   <FunctionBlock NumWords=61 BlockCodeSize=4>
// CHECK-7-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:     <Name abbrevid=5 op0=17/> blob data = 'anonPrivateMethod'
// CHECK-7-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-7-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:       <Name abbrevid=5 op0=9/> blob data = 'AnonClass'
// CHECK-7-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-7-NEXT:     </ReferenceBlock>
// CHECK-7-NEXT:     <ReferenceBlock NumWords=7 BlockCodeSize=4>
// CHECK-7-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-7-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-7-NEXT:     </ReferenceBlock>
// CHECK-7-NEXT:     <IsMethod abbrevid=9 op0=1/>
// CHECK-7-NEXT:     <Location abbrevid=7 op0=88 op1=4/> blob data = '{{.*}}'
// CHECK-7-NEXT:     <ReferenceBlock NumWords=12 BlockCodeSize=4>
// CHECK-7-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-7-NEXT:       <Name abbrevid=5 op0=9/> blob data = 'AnonClass'
// CHECK-7-NEXT:       <RefType abbrevid=6 op0=2/>
// CHECK-7-NEXT:       <Field abbrevid=7 op0=2/>
// CHECK-7-NEXT:     </ReferenceBlock>
// CHECK-7-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-7-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-7-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-7-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-7-NEXT:       </ReferenceBlock>
// CHECK-7-NEXT:     </TypeBlock>
// CHECK-7-NEXT:   </FunctionBlock>
// CHECK-7-NEXT: </RecordBlock>

// RUN: llvm-bcanalyzer --dump %t/docs/bc/83CC52D32583E0771710A7742DE81C839E953AC8.bc | FileCheck %s --check-prefix CHECK-8
// CHECK-8: <BLOCKINFO_BLOCK/>
// CHECK-8-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-8-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-8-NEXT: </VersionBlock>
// CHECK-8-NEXT: <NamespaceBlock NumWords=41 BlockCodeSize=4>
// CHECK-8-NEXT:   <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-8-NEXT:   <FunctionBlock NumWords=33 BlockCodeSize=4>
// CHECK-8-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-8-NEXT:     <Name abbrevid=5 op0=18/> blob data = 'anonInlineFunction'
// CHECK-8-NEXT:     <ReferenceBlock NumWords=7 BlockCodeSize=4>
// CHECK-8-NEXT:       <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-8-NEXT:       <RefType abbrevid=6 op0=1/>
// CHECK-8-NEXT:       <Field abbrevid=7 op0=1/>
// CHECK-8-NEXT:     </ReferenceBlock>
// CHECK-8-NEXT:     <Location abbrevid=7 op0=94 op1=4/> blob data = '{{.*}}'
// CHECK-8-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-8-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-8-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-8-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-8-NEXT:       </ReferenceBlock>
// CHECK-8-NEXT:     </TypeBlock>
// CHECK-8-NEXT:   </FunctionBlock>
// CHECK-8-NEXT: </NamespaceBlock>
