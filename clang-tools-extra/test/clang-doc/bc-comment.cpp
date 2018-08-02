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

// RUN: clang-doc --dump-intermediate --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: llvm-bcanalyzer --dump %t/docs/bc/0000000000000000000000000000000000000000.bc | FileCheck %s --check-prefix CHECK-0
// CHECK-0: <BLOCKINFO_BLOCK/>
// CHECK-0-NEXT: <VersionBlock NumWords=1 BlockCodeSize=4>
// CHECK-0-NEXT:   <Version abbrevid=4 op0=2/>
// CHECK-0-NEXT: </VersionBlock>
// CHECK-0-NEXT: <NamespaceBlock NumWords=432 BlockCodeSize=4>
// CHECK-0-NEXT:   <FunctionBlock NumWords=429 BlockCodeSize=4>
// CHECK-0-NEXT:     <USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>
// CHECK-0-NEXT:     <Name abbrevid=5 op0=1/> blob data = 'F'
// CHECK-0-NEXT:     <CommentBlock NumWords=354 BlockCodeSize=4>
// CHECK-0-NEXT:       <Kind abbrevid=4 op0=11/> blob data = 'FullComment'
// CHECK-0-NEXT:       <CommentBlock NumWords=13 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=31 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=19/> blob data = 'BlockCommandComment'
// CHECK-0-NEXT:         <Name abbrevid=6 op0=5/> blob data = 'brief'
// CHECK-0-NEXT:         <CommentBlock NumWords=19 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:           <CommentBlock NumWords=11 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:             <Text abbrevid=5 op0=19/> blob data = ' Brief description.'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=37 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=13 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=26/> blob data = ' Extended description that'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=14 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=30/> blob data = ' continues onto the next line.'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=76 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=14 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=19/> blob data = 'HTMLStartTagComment'
// CHECK-0-NEXT:           <Name abbrevid=6 op0=2/> blob data = 'ul'
// CHECK-0-NEXT:           <AttrKey abbrevid=12 op0=5/> blob data = 'class'
// CHECK-0-NEXT:           <AttrVal abbrevid=13 op0=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=9 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=19/> blob data = 'HTMLStartTagComment'
// CHECK-0-NEXT:           <Name abbrevid=6 op0=2/> blob data = 'li'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=9 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=9/> blob data = ' Testing.'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=9 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=17/> blob data = 'HTMLEndTagComment'
// CHECK-0-NEXT:           <Name abbrevid=6 op0=2/> blob data = 'ul'
// CHECK-0-NEXT:           <SelfClosing abbrevid=10 op0=1/>
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=13 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=32 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=20/> blob data = 'VerbatimBlockComment'
// CHECK-0-NEXT:         <Name abbrevid=6 op0=8/> blob data = 'verbatim'
// CHECK-0-NEXT:         <CloseName abbrevid=9 op0=11/> blob data = 'endverbatim'
// CHECK-0-NEXT:         <CommentBlock NumWords=16 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=24/> blob data = 'VerbatimBlockLineComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=27/> blob data = ' The description continues.'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=22 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=7 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=3/> blob data = ' --'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:         <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=39 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=19/> blob data = 'ParamCommandComment'
// CHECK-0-NEXT:         <Direction abbrevid=7 op0=5/> blob data = '[out]'
// CHECK-0-NEXT:         <ParamName abbrevid=8 op0=1/> blob data = 'I'
// CHECK-0-NEXT:         <Explicit abbrevid=11 op0=1/>
// CHECK-0-NEXT:         <CommentBlock NumWords=25 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:           <CommentBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:             <Text abbrevid=5 op0=16/> blob data = ' is a parameter.'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:           <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=38 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=19/> blob data = 'ParamCommandComment'
// CHECK-0-NEXT:         <Direction abbrevid=7 op0=4/> blob data = '[in]'
// CHECK-0-NEXT:         <ParamName abbrevid=8 op0=1/> blob data = 'J'
// CHECK-0-NEXT:         <CommentBlock NumWords=25 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:           <CommentBlock NumWords=10 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:             <Text abbrevid=5 op0=16/> blob data = ' is a parameter.'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:           <CommentBlock NumWords=5 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:       <CommentBlock NumWords=28 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=19/> blob data = 'BlockCommandComment'
// CHECK-0-NEXT:         <Name abbrevid=6 op0=6/> blob data = 'return'
// CHECK-0-NEXT:         <CommentBlock NumWords=16 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:           <CommentBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:             <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:             <Text abbrevid=5 op0=5/> blob data = ' void'
// CHECK-0-NEXT:           </CommentBlock>
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:     </CommentBlock>
// CHECK-0-NEXT:     <CommentBlock NumWords=28 BlockCodeSize=4>
// CHECK-0-NEXT:       <Kind abbrevid=4 op0=11/> blob data = 'FullComment'
// CHECK-0-NEXT:       <CommentBlock NumWords=21 BlockCodeSize=4>
// CHECK-0-NEXT:         <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
// CHECK-0-NEXT:         <CommentBlock NumWords=13 BlockCodeSize=4>
// CHECK-0-NEXT:           <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
// CHECK-0-NEXT:           <Text abbrevid=5 op0=28/> blob data = ' Bonus comment on definition'
// CHECK-0-NEXT:         </CommentBlock>
// CHECK-0-NEXT:       </CommentBlock>
// CHECK-0-NEXT:     </CommentBlock>
// CHECK-0-NEXT:     <DefLocation abbrevid=6 op0=28 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <Location abbrevid=7 op0=25 op1=4/> blob data = '{{.*}}'
// CHECK-0-NEXT:     <TypeBlock NumWords=6 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=4/> blob data = 'void'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:     </TypeBlock>
// CHECK-0-NEXT:     <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'I'
// CHECK-0-NEXT:     </FieldTypeBlock>
// CHECK-0-NEXT:     <FieldTypeBlock NumWords=8 BlockCodeSize=4>
// CHECK-0-NEXT:       <ReferenceBlock NumWords=3 BlockCodeSize=4>
// CHECK-0-NEXT:         <Name abbrevid=5 op0=3/> blob data = 'int'
// CHECK-0-NEXT:         <Field abbrevid=7 op0=4/>
// CHECK-0-NEXT:       </ReferenceBlock>
// CHECK-0-NEXT:       <Name abbrevid=4 op0=1/> blob data = 'J'
// CHECK-0-NEXT:     </FieldTypeBlock>
// CHECK-0-NEXT:   </FunctionBlock>
// CHECK-0-NEXT: </NamespaceBlock>
