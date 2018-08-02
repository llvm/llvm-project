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

// RUN: clang-doc --format=yaml --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./F.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '7574630614A535710E5A6ABCFFF98BCA2D06A4CA'
// CHECK-0-NEXT: Name:            'F'
// CHECK-0-NEXT: Description:     
// CHECK-0-NEXT:   - Kind:            'FullComment'
// CHECK-0-NEXT:     Children:        
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:       - Kind:            'BlockCommandComment'
// CHECK-0-NEXT:         Name:            'brief'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'ParagraphComment'
// CHECK-0-NEXT:             Children:        
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:                 Text:            ' Brief description.'
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:             Text:            ' Extended description that'
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:             Text:            ' continues onto the next line.'
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:           - Kind:            'HTMLStartTagComment'
// CHECK-0-NEXT:             Name:            'ul'
// CHECK-0-NEXT:             AttrKeys:        
// CHECK-0-NEXT:               - 'class'
// CHECK-0-NEXT:             AttrValues:      
// CHECK-0-NEXT:               - 'test'
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:           - Kind:            'HTMLStartTagComment'
// CHECK-0-NEXT:             Name:            'li'
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:             Text:            ' Testing.'
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:           - Kind:            'HTMLEndTagComment'
// CHECK-0-NEXT:             Name:            'ul'
// CHECK-0-NEXT:             SelfClosing:     true
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:       - Kind:            'VerbatimBlockComment'
// CHECK-0-NEXT:         Name:            'verbatim'
// CHECK-0-NEXT:         CloseName:       'endverbatim'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'VerbatimBlockLineComment'
// CHECK-0-NEXT:             Text:            ' The description continues.'
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:             Text:            ' --'
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:       - Kind:            'ParamCommandComment'
// CHECK-0-NEXT:         Direction:       '[out]'
// CHECK-0-NEXT:         ParamName:       'I'
// CHECK-0-NEXT:         Explicit:        true
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'ParagraphComment'
// CHECK-0-NEXT:             Children:        
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:                 Text:            ' is a parameter.'
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:       - Kind:            'ParamCommandComment'
// CHECK-0-NEXT:         Direction:       '[in]'
// CHECK-0-NEXT:         ParamName:       'J'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'ParagraphComment'
// CHECK-0-NEXT:             Children:        
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:                 Text:            ' is a parameter.'
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:       - Kind:            'BlockCommandComment'
// CHECK-0-NEXT:         Name:            'return'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'ParagraphComment'
// CHECK-0-NEXT:             Children:        
// CHECK-0-NEXT:               - Kind:            'TextComment'
// CHECK-0-NEXT:                 Text:            ' void'
// CHECK-0-NEXT:   - Kind:            'FullComment'
// CHECK-0-NEXT:     Children:        
// CHECK-0-NEXT:       - Kind:            'ParagraphComment'
// CHECK-0-NEXT:         Children:        
// CHECK-0-NEXT:           - Kind:            'TextComment'
// CHECK-0-NEXT:             Text:            ' Bonus comment on definition'
// CHECK-0-NEXT: DefLocation:     
// CHECK-0-NEXT:   LineNumber:      28
// CHECK-0-NEXT:   Filename:        'test'
// CHECK-0-NEXT: Location:        
// CHECK-0-NEXT:   - LineNumber:      25
// CHECK-0-NEXT:     Filename:        'test'
// CHECK-0-NEXT: Params:          
// CHECK-0-NEXT:   - Type:            
// CHECK-0-NEXT:       Name:            'int'
// CHECK-0-NEXT:     Name:            'I'
// CHECK-0-NEXT:   - Type:            
// CHECK-0-NEXT:       Name:            'int'
// CHECK-0-NEXT:     Name:            'J'
// CHECK-0-NEXT: ReturnType:      
// CHECK-0-NEXT:   Type:            
// CHECK-0-NEXT:     Name:            'void'
// CHECK-0-NEXT: ...
