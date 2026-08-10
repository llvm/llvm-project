// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng std=c++11 %s > %t/out
// RUN: FileCheck %s < %t/out

namespace inner {
  //! This documentation should be inherited.
  struct Opaque;
}
// CHECK:         (CXComment_Text Text=[ This documentation should be inherited.])))] 

namespace borrow {
  //! This is documentation for the typedef (which shows up).
  typedef inner::Opaque Typedef;
// CHECK:         (CXComment_Text Text=[ This is documentation for the typedef (which shows up).])))] 

  //! This is documentation for the alias (which shows up).
  using Alias = inner::Opaque;
// CHECK:         (CXComment_Text Text=[ This is documentation for the alias (which shows up).])))] 

  typedef inner::Opaque NoDocTypedef;
// CHECK:         (CXComment_Text Text=[ This documentation should be inherited.])))] 

  using NoDocAlias = inner::Opaque;
// CHECK:         (CXComment_Text Text=[ This documentation should be inherited.])))] 
}
