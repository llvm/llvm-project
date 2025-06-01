// RUN: clang-doc --public --format=yaml -p %T %s -output=%t
// RUN: FileCheck --input-file=%S/Inputs/html-tag-comment.yaml %s

/// \verbatim <ul class="test"><li> Testing. </li></ul> \endverbatim
void withHtmlTag() {}
// CHECK: ---
// CHECK: Name:            'withHtmlTag'
// CHECK: Description:
// CHECK:   - Kind:          FullComment
// CHECK:     Children:
// CHECK:       - Kind:        VerbatimBlockComment
// CHECK:         Name:        'verbatim'
// CHECK:         CloseName:   'endverbatim'
// CHECK:         Children:
// CHECK:           - Kind:      VerbatimBlockLineComment
// CHECK:             Text:      '<ul class="test"><li> Testing. </li></ul>'
// CHECK: ...
