// REQUIRES: xmllint

// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-other-01.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-01.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-02.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-03.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-04.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-05.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-06.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-07.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-08.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-09.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-function-10.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-availability-attr-01.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-availability-attr-02.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-deprecated-attr.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-unavailable-attr.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-class-01.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-class-02.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-class-03.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-class-04.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-variable-01.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-namespace-01.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-typedef-01.xml
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-typedef-02.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-enum-01.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-para-kind-01.xml
//
// RUN: xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/valid-inline-command-01.xml

// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-01.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-02.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-03.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-04.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-05.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-06.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-07.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-08.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-09.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-10.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-11.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-12.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-function-13.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
//
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-para-kind-01.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// RUN: not xmllint --noout --relaxng %S/../../bindings/xml/comment-xml-schema.rng %S/Inputs/CommentXML/invalid-para-kind-02.xml 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID

// CHECK-INVALID: fails to validate

