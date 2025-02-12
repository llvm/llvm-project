// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t/Headers
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %t/Headers -fexperimental-bounds-safety-attributes %t/Headers/SemaErrors.c 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %t/Headers -fexperimental-bounds-safety-attributes %t/Headers/NegLevel.c 2>&1 | FileCheck %s --check-prefix NEGLEVEL
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %t/Headers -fexperimental-bounds-safety-attributes %t/Headers/InvalidKind.c 2>&1 | FileCheck %s --check-prefix INVALIDKIND

//--- module.modulemap
module SemaErrors {
  header "SemaErrors.h"
}
module NegLevel {
  header "NegLevel.h"
}
module InvalidKind {
  header "InvalidKind.h"
}

//--- SemaErrors.c
#include "SemaErrors.h"
//--- SemaErrors.apinotes
Name: OOBLevel
Functions:
  - Name:              oob_level
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 42
            BoundedBy: len
  - Name:              off_by_1_level
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 1
            BoundedBy: len
  - Name:              nonpointer_param
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 0
            BoundedBy: len
  - Name:              wrong_name
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 0
            BoundedBy: len
  - Name:              wrong_scope
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 0
            BoundedBy: static_len
  - Name:              mismatching_count
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: 0
            BoundedBy: apinote_len
  - Name:              mismatching_end
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: ended_by
            Level: 0
            BoundedBy: apinote_end
//--- SemaErrors.h
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: __counted_by attribute only applies to pointer arguments
// CHECK-NEXT: oob_level
void oob_level(int * buf, int len);
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: __counted_by attribute only applies to pointer arguments
// CHECK-NEXT: off_by_1_level
void off_by_1_level(int * buf, int len);
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: __counted_by attribute only applies to pointer arguments
// CHECK-NEXT: nonpointer_param
void nonpointer_param(int buf, int len);
// CHECK: <API Notes>:1:1: error: use of undeclared identifier 'len'; did you mean 'len2'?
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: note: 'len2' declared here
// CHECK-NEXT: wrong_name
void wrong_name(int * buf, int len2);
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: count expression in function declaration may only reference parameters of that function
// CHECK-NEXT: wrong_scope
int static_len = 5;
void wrong_scope(int * buf);
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: pointer cannot have more than one count attribute
// CHECK-NEXT: mismatching_count
void mismatching_count(int * __attribute__((__counted_by__(header_len))) buf, int apinote_len, int header_len);
// CHECK: SemaErrors.h:{{.*}}:{{.*}}: error: pointer cannot have more than one end attribute
// CHECK-NEXT: mismatching_end
void mismatching_end(int * __attribute__((__ended_by__(header_end))) buf, int * apinote_end, int * header_end);
// CHECK: SemaErrors.c:{{.*}}:{{.*}}: fatal error: could not build module 'SemaErrors'


//--- NegLevel.apinotes
Name: NegLevel
Functions:
  - Name:              neg_level
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: counted_by
            Level: -1
            BoundedBy: len
//--- NegLevel.h
void neg_level(int * buf, int len);
//--- NegLevel.c
#include "NegLevel.h"
// NEGLEVEL: NegLevel.apinotes:{{.*}}:{{.*}}: error: invalid number
// NEGLEVEL-NEXT: Level: -1
// NEGLEVEL: NegLevel.c:{{.*}}:{{.*}}: fatal error: could not build module 'NegLevel'


//--- InvalidKind.apinotes
Name: InvalidKind
Functions:
  - Name:              invalid_kind
    Parameters:
      - Position:      0
        BoundsSafety:
            Kind: __counted_by
            Level: 0
            BoundedBy: len
//--- InvalidKind.h
void invalid_kind(int * buf, int len);
//--- InvalidKind.c
#include "InvalidKind.h"
// INVALIDKIND: InvalidKind.apinotes:{{.*}}:{{.*}}: error: unknown enumerated scalar
// INVALIDKIND-NEXT: Kind: __counted_by
// INVALIDKIND: InvalidKind.c:{{.*}}:{{.*}}: fatal error: could not build module 'InvalidKind'

