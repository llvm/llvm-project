// RUN: clang-tblgen -gen-attr-docs -I%p/../../include %s -o - 2>&1 | FileCheck %s

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

// CHECK: //--- AttributeReference.md
// CHECK: :::{include} AttributeReference/TestAttributes.md
// CHECK-NEXT: :::
// CHECK: //--- AttributeReference/TestAttributes.md
// CHECK: ## Test Attributes
// CHECK: Test category introduction.
// CHECK: ### test_split_attr
// CHECK: `test_split_attr`
// CHECK: Test split documentation content.
