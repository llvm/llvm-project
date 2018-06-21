// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --dump-intermediate -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: llvm-bcanalyzer %t/docs/bc/7574630614A535710E5A6ABCFFF98BCA2D06A4CA.bc --dump | FileCheck %s
 
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

// CHECK: <BLOCKINFO_BLOCK/>
// CHECK-NEXT: <VersionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <Version abbrevid=4 op0=2/>
// CHECK-NEXT: </VersionBlock>
// CHECK-NEXT: <FunctionBlock NumWords={{[0-9]*}} BlockCodeSize=4>
  // CHECK-NEXT: <USR abbrevid=4 op0=20 op1=117 op2=116 op3=99 op4=6 op5=20 op6=165 op7=53 op8=113 op9=14 op10=90 op11=106 op12=188 op13=255 op14=249 op15=139 op16=202 op17=45 op18=6 op19=164 op20=202/>
  // CHECK-NEXT: <Name abbrevid=5 op0=1/> blob data = 'F'
  // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'FullComment'
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'BlockCommandComment'
      // CHECK-NEXT: <Name abbrevid=6 op0=5/> blob data = 'brief'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
          // CHECK-NEXT: <Text abbrevid=5 op0=19/> blob data = ' Brief description.'
        // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=26/> blob data = ' Extended description that'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=30/> blob data = ' continues onto the next line.'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'HTMLStartTagComment'
        // CHECK-NEXT: <Name abbrevid=6 op0=2/> blob data = 'ul'
        // CHECK-NEXT: <AttrKey abbrevid=12 op0=5/> blob data = 'class'
        // CHECK-NEXT: <AttrVal abbrevid=13 op0=4/> blob data = 'test'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'HTMLStartTagComment'
        // CHECK-NEXT: <Name abbrevid=6 op0=2/> blob data = 'li'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=9/> blob data = ' Testing.'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=17/> blob data = 'HTMLEndTagComment'
        // CHECK-NEXT: <Name abbrevid=6 op0=2/> blob data = 'ul'
        // CHECK-NEXT: <SelfClosing abbrevid=10 op0=1/>
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=20/> blob data = 'VerbatimBlockComment'
      // CHECK-NEXT: <Name abbrevid=6 op0=8/> blob data = 'verbatim'
      // CHECK-NEXT: <CloseName abbrevid=9 op0=11/> blob data = 'endverbatim'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=24/> blob data = 'VerbatimBlockLineComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=27/> blob data = ' The description continues.'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=3/> blob data = ' --'
      // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'ParamCommandComment'
      // CHECK-NEXT: <Direction abbrevid=7 op0=5/> blob data = '[out]'
      // CHECK-NEXT: <ParamName abbrevid=8 op0=1/> blob data = 'I'
      // CHECK-NEXT: <Explicit abbrevid=11 op0=1/>
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
          // CHECK-NEXT: <Text abbrevid=5 op0=16/> blob data = ' is a parameter.'
        // CHECK-NEXT: </CommentBlock>
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'ParamCommandComment'
      // CHECK-NEXT: <Direction abbrevid=7 op0=4/> blob data = '[in]'
      // CHECK-NEXT: <ParamName abbrevid=8 op0=1/> blob data = 'J'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
          // CHECK-NEXT: <Text abbrevid=5 op0=16/> blob data = ' is a parameter.'
        // CHECK-NEXT: </CommentBlock>
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=19/> blob data = 'BlockCommandComment'
      // CHECK-NEXT: <Name abbrevid=6 op0=6/> blob data = 'return'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
        // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
          // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
          // CHECK-NEXT: <Text abbrevid=5 op0=5/> blob data = ' void'
        // CHECK-NEXT: </CommentBlock>
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
  // CHECK-NEXT: </CommentBlock>
  // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'FullComment'
    // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Kind abbrevid=4 op0=16/> blob data = 'ParagraphComment'
      // CHECK-NEXT: <CommentBlock NumWords={{[0-9]*}} BlockCodeSize=4>
        // CHECK-NEXT: <Kind abbrevid=4 op0=11/> blob data = 'TextComment'
        // CHECK-NEXT: <Text abbrevid=5 op0=28/> blob data = ' Bonus comment on definition'
      // CHECK-NEXT: </CommentBlock>
    // CHECK-NEXT: </CommentBlock>
  // CHECK-NEXT: </CommentBlock>
  // CHECK-NEXT: <DefLocation abbrevid=6 op0=27 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <Location abbrevid=7 op0=24 op1={{[0-9]*}}/> blob data = '{{.*}}'
  // CHECK-NEXT: <TypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Name abbrevid=5 op0=4/> blob data = 'void'
      // CHECK-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-NEXT: </ReferenceBlock>
  // CHECK-NEXT: </TypeBlock>
  // CHECK-NEXT: <FieldTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-NEXT: </ReferenceBlock>
    // CHECK-NEXT: <Name abbrevid=4 op0=1/> blob data = 'I'
  // CHECK-NEXT: </FieldTypeBlock>
  // CHECK-NEXT: <FieldTypeBlock NumWords={{[0-9]*}} BlockCodeSize=4>
    // CHECK-NEXT: <ReferenceBlock NumWords={{[0-9]*}} BlockCodeSize=4>
      // CHECK-NEXT: <Name abbrevid=5 op0=3/> blob data = 'int'
      // CHECK-NEXT: <Field abbrevid=7 op0=4/>
    // CHECK-NEXT: </ReferenceBlock>
    // CHECK-NEXT: <Name abbrevid=4 op0=1/> blob data = 'J'
  // CHECK-NEXT: </FieldTypeBlock>
// CHECK-NEXT: </FunctionBlock>
