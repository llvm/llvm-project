// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-mapper -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/F0F9FC65FC90F54F690144A7AFB15DFC3D69B6E6.bc --dump | FileCheck %s --check-prefix CHECK-G-F
// RUN: llvm-bcanalyzer %t/docs/bc/4202E8BF0ECB12AE354C8499C52725B0EE30AED5.bc --dump | FileCheck %s --check-prefix CHECK-G

class G {
public: 
	int Method(int param) { return param; }
};

// CHECK-G: <BLOCKINFO_BLOCK/>
// CHECK-G-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-G-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-G-NEXT: </VersionBlock>
// CHECK-G-NEXT: <RecordBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-G-NEXT: <USR abbrevid=4 op0=20 op1=66 op2=2 op3=232 op4=191 op5=14 op6=203 op7=18 op8=174 op9=53 op10=76 op11=132 op12=153 op13=197 op14=39 op15=37 op16=176 op17=238 op18=48 op19=174 op20=213/>
  // CHECK-G-NEXT: <Name abbrevid=5 op0=1/> blob data = 'G'
  // CHECK-G-NEXT: <DefLocation abbrevid=7 op0=9 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-G-NEXT: <TagType abbrevid=9 op0=3/>
// CHECK-G-NEXT: </RecordBlock>

// CHECK-G-F: <BLOCKINFO_BLOCK/>
// CHECK-G-F-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
  // CHECK-G-F-NEXT: <Version abbrevid=4 op0=1/>
// CHECK-G-F-NEXT: </VersionBlock>
// CHECK-G-F-NEXT: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-G-F-NEXT: <USR abbrevid=4 op0=20 op1=240 op2=249 op3=252 op4=101 op5=252 op6=144 op7=245 op8=79 op9=105 op10=1 op11=68 op12=167 op13=175 op14=177 op15=93 op16=252 op17=61 op18=105 op19=182 op20=230/>
  // CHECK-G-F-NEXT: <Name abbrevid=5 op0=6/> blob data = 'Method'
  // CHECK-G-F-NEXT: <Namespace abbrevid=6 op0=1 op1=40/> blob data = '4202E8BF0ECB12AE354C8499C52725B0EE30AED5'
  // CHECK-G-F-NEXT: <IsMethod abbrevid=11 op0=1/>
  // CHECK-G-F-NEXT: <DefLocation abbrevid=7 op0=11 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-G-F-NEXT: <Parent abbrevid=9 op0=1 op1=40/> blob data = '4202E8BF0ECB12AE354C8499C52725B0EE30AED5'
  // CHECK-G-F-NEXT: <TypeBlock NumWords=4 BlockCodeSize=4>
    // CHECK-G-F-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
  // CHECK-G-F-NEXT: </TypeBlock>
  // CHECK-G-F-NEXT: <FieldTypeBlock NumWords=7 BlockCodeSize=4>
    // CHECK-G-F-NEXT: <Type abbrevid=4 op0=4 op1=3/> blob data = 'int'
    // CHECK-G-F-NEXT: <Name abbrevid=5 op0=5/> blob data = 'param'
  // CHECK-G-F-NEXT: </FieldTypeBlock>
// CHECK-G-F-NEXT: </FunctionBlock>
