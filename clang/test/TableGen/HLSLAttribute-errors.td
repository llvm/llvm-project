// RUN: not clang-tblgen -gen-clang-attr-parsed-attr-kinds -I%p/../../include %s -o - 2>&1 | FileCheck %s

include "clang/Basic/Attr.td"

// CHECK: error: HLSLAnnotation Attribute must be lower case.
def HLSLSV_FAKE: HLSLAnnotationAttr {
  let Spellings = [HLSLAnnotation<"SV_Fake">];
  let Subjects = SubjectList<[ParmVar, Field]>;
  let LangOpts = [HLSL];
  let Documentation = [HLSLSV_GroupThreadIDDocs];
}
