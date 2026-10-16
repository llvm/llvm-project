// RUN: clang-tblgen -gen-attr-docs -I%p/../../include %s -o - 2>&1 | FileCheck %s --check-prefix=DOCS
// RUN: clang-tblgen -gen-attr-doc-syntaxes -I%p/../../include %s -o - 2>&1 | FileCheck %s --check-prefix=SYNTAX

include "clang/Basic/Attr.td"

def TestSplitDocCategory : DocumentationCategory<"Test Attributes"> {
  let Content = [{
Test category introduction.
}];
}

def TestSplitDocs : Documentation {
  let Category = TestSplitDocCategory;
  let Content = [{
Test split documentation content.
}];
}

def TestSplitAttr : InheritableAttr {
  let Spellings = [Clang<"test_split_attr">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [TestSplitDocs];
}

// DOCS: //--- AttributeReference.md
// DOCS: :::{include} AttributeReference/TestAttributes.md
// DOCS-NEXT: :::
// DOCS: //--- AttributeReference/TestAttributes.md
// DOCS: ## Test Attributes
// DOCS: Test category introduction.
// DOCS: ### test_split_attr
// DOCS: {clang-attr-syntaxes}`TestSplitDocs`
// DOCS: Test split documentation content.

// SYNTAX: "doc": "TestSplitDocs"
// SYNTAX: "heading": "test_split_attr"
// SYNTAX: "category": "Test Attributes"
// SYNTAX: "attributes": [
// SYNTAX-NEXT: "TestSplitAttr"
// SYNTAX: "name": "GNU"
// SYNTAX: "spellings": [
// SYNTAX-NEXT: "test_split_attr"
