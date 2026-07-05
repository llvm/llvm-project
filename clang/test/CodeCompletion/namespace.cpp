namespace N3 {
}

namespace N2 {
  namespace I1 { }
  namespace I4 = I1;
  namespace I5 { }
  namespace I1 { }
  
  namespace 
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):13 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: I1
  // CHECK-CC1-NEXT: I5
  
