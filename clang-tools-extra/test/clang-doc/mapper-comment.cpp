// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

/// \brief Brief description.
///
/// Extended description that
/// continues onto the next line.
/// 
/// <ul class="test">
///   <li> Testing.
/// </ul>
///
/// \verbatim
/// The description continues.
/// \endverbatim
/// --
/// \param [out] I is a parameter.
/// \param J is a parameter.
/// \return void
void F(int I, int J);

/// Bonus comment on definition
void F(int I, int J) {}

// RUN: clang-doc --dump-mapper --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/7574630614A535710E5A6ABCFFF98BCA2D06A4CA.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <FunctionBlock NumWords=70 BlockCodeSize=4>
// CHECK-0-NEXT:   <USR abbrevid=4 op0=20 op1=117 op2=116 op3=99 op4=6 op5=20 op6=165 op7=53 op8=113 op9=14 op10=90 op11=106 op12=188 op13=255 op14=249 op15=139 op16=202 op17=45 op18=6 op19=164 op20=202/>
// CHECK-0-NEXT:   <Name abbrevid=5 op0=1/> blob data = 'F'
// CHECK-0-NEXT:   <CommentBlock NumWords=28 BlockCodeSize=4>
// CHECK-0-NEXT:     <Kind abbrevid=4 op0=11/> blob data = 'FullComment'
// CHECK-0-NEXT:     <CommentBlock NumWords=21 BlockCodeSize=4>
// CHECK-0-NEXT:       <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:       <CommentBlock NumWords=13 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         <Text abbrevid=5 op0=28/> blob data = ' Bonus comment on definition'
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:     </CommentBlock>
// CHECK-0-NEXT:   </CommentBlock>
// CHECK-0-NEXT:   <DefLocation abbrevid=6 op0=28 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:   <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-0-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-0-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:   </TypeBlock>
// CHECK-0-NEXT:   <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'I'
// CHECK-0-NEXT:   </FieldTypeBlock>
// CHECK-0-NEXT:   <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:     <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:       <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:       <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:     </ReferenceBlock>
// CHECK-0-NEXT:     <Name abbrevid=4 op0=1/> blob data = 'J'
// CHECK-0-NEXT:   </FieldTypeBlock>
// CHECK-0-NEXT: </FunctionBlock>
