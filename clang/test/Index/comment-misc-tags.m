// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out

/*!
     @interface IOCommandGate
     @brief    This is a brief
     @abstract Single-threaded work-loop client request mechanism.
     @discussion An IOCommandGate instance is an extremely light weight mechanism that
         executes an action on the driver's work-loop...
     @textblock
       Many discussions about text
       Many1 discussions about text
       Many2 discussions about text
     @/textblock
     @link //un_ref/c/func/function_name link text goes here @/link
     @see  //un_ref/doc/uid/XX0000011 I/O Kit Fundamentals
     @seealso //k_ref/doc/uid/XX30000905-CH204 Programming
 */
@interface IOCommandGate
@end

// CHECK:       (CXComment_BlockCommand CommandName=[abstract]
// CHECK-NEXT:    (CXComment_Paragraph
// CHECK-NEXT:       (CXComment_Text Text=[ Single-threaded work-loop client request mechanism.] HasTrailingNewline)
// CHECK:       (CXComment_BlockCommand CommandName=[discussion]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:       (CXComment_Text Text=[ An IOCommandGate instance is an extremely light weight mechanism that] HasTrailingNewline)
// CHECK-NEXT:       (CXComment_Text Text=[         executes an action on the driver's work-loop...] HasTrailingNewline)
// CHECK:       (CXComment_VerbatimBlockCommand CommandName=[textblock]
// CHECK-NEXT:     (CXComment_VerbatimBlockLine Text=[       Many discussions about text])
// CHECK-NEXT:       (CXComment_VerbatimBlockLine Text=[       Many1 discussions about text])
// CHECK-NEXT:       (CXComment_VerbatimBlockLine Text=[       Many2 discussions about text]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace

// CHECK:       (CXComment_VerbatimBlockCommand CommandName=[link]
// CHECK-NEXT:     (CXComment_VerbatimBlockLine Text=[ //un_ref/c/func/function_name link text goes here ]))
// CHECK-NEXT:     (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:     (CXComment_Text Text=[     ] IsWhitespace))
// CHECK:       (CXComment_BlockCommand CommandName=[see]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:     (CXComment_Text Text=[  //un_ref/doc/uid/XX0000011 I/O Kit Fundamentals] HasTrailingNewline)
// CHECK-NEXT:     (CXComment_Text Text=[     ] IsWhitespace)))
// CHECK:       (CXComment_BlockCommand CommandName=[seealso]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:     (CXComment_Text Text=[ //k_ref/doc/uid/XX30000905-CH204 Programming])

/*!
\arg \c AlignLeft left alignment.
\li \c AlignRight right alignment.

  No other types of alignment are supported.
*/
struct S {
  int AlignLeft;
  int AlignRight;
};

// CHECK:       (CXComment_BlockCommand CommandName=[arg]
// CHECK-NEXT:    (CXComment_Paragraph
// CHECK-NEXT:    (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:    (CXComment_InlineCommand CommandName=[c] RenderMonospaced Arg[0]=AlignLeft)
// CHECK-NEXT:    (CXComment_Text Text=[ left alignment.] HasTrailingNewline)))
// CHECK:       (CXComment_BlockCommand CommandName=[li]
// CHECK-NEXT:    (CXComment_Paragraph
// CHECK-NEXT:    (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:    (CXComment_InlineCommand CommandName=[c] RenderMonospaced Arg[0]=AlignRight)
// CHECK-NEXT:    (CXComment_Text Text=[ right alignment.])))
// CHECK:       (CXComment_Paragraph
// CHECK-NEXT:    (CXComment_Text Text=[  No other types of alignment are supported.]))

/*! \struct Test
 * Normal text.
 *
 * \par User defined paragraph:
 * Contents of the paragraph.
 *
 * \par
 * New paragraph under the same heading.
 *
 * \note
 * This note consists of two paragraphs.
 * This is the first paragraph.
 *
 * \par
 * And this is the second paragraph.
 *
 * More normal text.
 */
  
struct Test {int filler;};

// CHECK:       (CXComment_BlockCommand CommandName=[par]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:        (CXComment_Text Text=[ User defined paragraph:] HasTrailingNewline)
// CHECK-NEXT:        (CXComment_Text Text=[ Contents of the paragraph.])))
// CHECK:       (CXComment_BlockCommand CommandName=[par]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:        (CXComment_Text Text=[ New paragraph under the same heading.])))
// CHECK:       (CXComment_BlockCommand CommandName=[note]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:        (CXComment_Text Text=[ This note consists of two paragraphs.] HasTrailingNewline)
// CHECK-NEXT:        (CXComment_Text Text=[ This is the first paragraph.])))
// CHECK:       (CXComment_BlockCommand CommandName=[par]
// CHECK-NEXT:     (CXComment_Paragraph
// CHECK-NEXT:     (CXComment_Text Text=[ And this is the second paragraph.])))

