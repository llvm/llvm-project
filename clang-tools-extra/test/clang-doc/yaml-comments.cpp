// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/F.yaml | FileCheck %s

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
///
/// \param [out] I is a parameter.
/// \param J is a parameter.
/// \return void
void F(int I, int J);

/// Bonus comment on definition
void F(int I, int J) {}

// CHECK: ---
// CHECK-NEXT: USR:             '7574630614A535710E5A6ABCFFF98BCA2D06A4CA'
// CHECK-NEXT: Name:            'F'
// CHECK-NEXT: Description:     
// CHECK-NEXT:   - Kind:            'FullComment'
// CHECK-NEXT:     Children:        
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:       - Kind:            'BlockCommandComment'
// CHECK-NEXT:         Name:            'brief'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'ParagraphComment'
// CHECK-NEXT:             Children:        
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:                 Text:            ' Brief description.'
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:             Text:            ' Extended description that'
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:             Text:            ' continues onto the next line.'
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:           - Kind:            'HTMLStartTagComment'
// CHECK-NEXT:             Name:            'ul'
// CHECK-NEXT:             AttrKeys:        
// CHECK-NEXT:               - 'class'
// CHECK-NEXT:             AttrValues:      
// CHECK-NEXT:               - 'test'
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:           - Kind:            'HTMLStartTagComment'
// CHECK-NEXT:             Name:            'li'
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:             Text:            ' Testing.'
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:           - Kind:            'HTMLEndTagComment'
// CHECK-NEXT:             Name:            'ul'
// CHECK-NEXT:             SelfClosing:     true
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:       - Kind:            'VerbatimBlockComment'
// CHECK-NEXT:         Name:            'verbatim'
// CHECK-NEXT:         CloseName:       'endverbatim'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'VerbatimBlockLineComment'
// CHECK-NEXT:             Text:            ' The description continues.'
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:       - Kind:            'ParamCommandComment'
// CHECK-NEXT:         Direction:       '[out]'
// CHECK-NEXT:         ParamName:       'I'
// CHECK-NEXT:         Explicit:        true
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'ParagraphComment'
// CHECK-NEXT:             Children:        
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:                 Text:            ' is a parameter.'
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:       - Kind:            'ParamCommandComment'
// CHECK-NEXT:         Direction:       '[in]'
// CHECK-NEXT:         ParamName:       'J'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'ParagraphComment'
// CHECK-NEXT:             Children:        
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:                 Text:            ' is a parameter.'
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:       - Kind:            'BlockCommandComment'
// CHECK-NEXT:         Name:            'return'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'ParagraphComment'
// CHECK-NEXT:             Children:        
// CHECK-NEXT:               - Kind:            'TextComment'
// CHECK-NEXT:                 Text:            ' void'
// CHECK-NEXT:   - Kind:            'FullComment'
// CHECK-NEXT:     Children:        
// CHECK-NEXT:       - Kind:            'ParagraphComment'
// CHECK-NEXT:         Children:        
// CHECK-NEXT:           - Kind:            'TextComment'
// CHECK-NEXT:             Text:            ' Bonus comment on definition'
// CHECK-NEXT: DefLocation:     
// CHECK-NEXT:   LineNumber:      27
// CHECK-NEXT:   Filename:        '{{.*}}'
// CHECK-NEXT: Location:        
// CHECK-NEXT:   - LineNumber:      24
// CHECK-NEXT:     Filename:        '{{.*}}'
// CHECK-NEXT: Params:          
// CHECK-NEXT:   - Type:            
// CHECK-NEXT:       Name:            'int'
// CHECK-NEXT:     Name:            'I'
// CHECK-NEXT:   - Type:            
// CHECK-NEXT:       Name:            'int'
// CHECK-NEXT:     Name:            'J'
// CHECK-NEXT: ReturnType:      
// CHECK-NEXT:   Type:            
// CHECK-NEXT:     Name:            'void'
// CHECK-NEXT: ...
