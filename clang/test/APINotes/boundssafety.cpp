// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks -fexperimental-bounds-safety-attributes %s -ast-dump -ast-dump-filter asdf | FileCheck %s --check-prefixes=CHECK,CHECK-MEMBER
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks -fexperimental-bounds-safety-attributes %s -ast-dump -ast-dump-filter asdf -DFREE_FUNCTIONS | FileCheck %s

#ifdef FREE_FUNCTIONS
#include "BoundsUnsafe.h"
#else
#include "BoundsUnsafeCxx.hpp"
#endif

// CHECK: asdf_counted 'void (int * __counted_by(len), int)'
// CHECK: buf 'int * __counted_by(len)':'int *'

// CHECK: asdf_sized 'void (int * __sized_by(size), int)'
// CHECK: buf 'int * __sized_by(size)':'int *'

// CHECK: asdf_counted_n 'void (int * __counted_by_or_null(len), int)'
// CHECK: buf 'int * __counted_by_or_null(len)':'int *'

// CHECK: asdf_sized_n 'void (int * __sized_by_or_null(size), int)'
// CHECK: buf 'int * __sized_by_or_null(size)':'int *'

// CHECK: asdf_ended 'void (int * __ended_by(end), int * /* __started_by(buf) */ )'
// CHECK: buf 'int * __ended_by(end)':'int *'
// CHECK: end 'int * /* __started_by(buf) */ ':'int *'

// CHECK: asdf_sized_mul 'void (int * __sized_by(size * count), int, int)'
// CHECK: buf 'int * __sized_by(size * count)':'int *'

// CHECK: asdf_counted_out 'void (int * __counted_by(*len)*, int *)'
// CHECK: buf 'int * __counted_by(*len)*'

// CHECK: asdf_counted_const 'void (int * __counted_by(7))'
// CHECK: buf 'int * __counted_by(7)':'int *'

// CHECK: asdf_counted_nullable 'void (int, int * __counted_by(len) _Nullable)'
// CHECK: buf 'int * __counted_by(len) _Nullable':'int *'

// CHECK: asdf_counted_noescape 'void (int * __counted_by(len), int)'
// CHECK: buf 'int * __counted_by(len)':'int *'
// CHECK-MEMBER-NEXT: | |-SwiftVersionedAdditionAttr {{.*}} Implicit 0 IsReplacedByActive
// CHECK-MEMBER-NEXT: | | `-NoEscapeAttr
// CHECK-NEXT: | `-NoEscapeAttr

// CHECK: asdf_counted_default_level 'void (int * __counted_by(len), int)'
// CHECK: buf 'int * __counted_by(len)':'int *'

// CHECK: asdf_counted_redundant 'void (int * __counted_by(len), int)'
// CHECK: buf 'int * __counted_by(len)':'int *'

// CHECK: asdf_ended_chained 'void (int * __ended_by(mid), int * __ended_by(end) /* __started_by(buf) */ , int * /* __started_by(mid) */ )'
// CHECK: buf 'int * __ended_by(mid)':'int *'
// CHECK: mid 'int * __ended_by(end) /* __started_by(buf) */ ':'int *'
// CHECK: end 'int * /* __started_by(mid) */ ':'int *'

// CHECK: asdf_ended_chained_reverse 'void (int * __ended_by(mid), int * __ended_by(end) /* __started_by(buf) */ , int * /* __started_by(mid) */ )'
// CHECK: buf 'int * __ended_by(mid)':'int *'
// CHECK: mid 'int * __ended_by(end) /* __started_by(buf) */ ':'int *'
// CHECK: end 'int * /* __started_by(mid) */ ':'int *'

// CHECK: asdf_ended_already_started 'void (int * __ended_by(mid), int * __ended_by(end) /* __started_by(buf) */ , int * /* __started_by(mid) */ )'
// CHECK: buf 'int * __ended_by(mid)':'int *'
// CHECK: mid 'int * __ended_by(end) /* __started_by(buf) */ ':'int *'
// CHECK: end 'int * /* __started_by(mid) */ ':'int *'

// CHECK: asdf_ended_already_ended 'void (int * __ended_by(mid), int * __ended_by(end) /* __started_by(buf) */ , int * /* __started_by(mid) */ )'
// CHECK: buf 'int * __ended_by(mid)':'int *'
// CHECK: mid 'int * __ended_by(end) /* __started_by(buf) */ ':'int *'
// CHECK: end 'int * /* __started_by(mid) */ ':'int *'

// CHECK: asdf_ended_redundant 'void (int * __ended_by(end), int * /* __started_by(buf) */ )'
// CHECK: buf 'int * __ended_by(end)':'int *'
// CHECK: end 'int * /* __started_by(buf) */ ':'int *'

// CHECK: asdf_nterm 'void (int * __terminated_by(0))'
// CHECK: buf 'int * __terminated_by(0)':'int *'


// CHECK: asdf_return_counted 'int * __counted_by(len)(int)'

// CHECK: asdf_return_sized 'int * __sized_by(size)(int)'

// CHECK: asdf_return_counted_n 'int * __counted_by_or_null(len)(int)'

// CHECK: asdf_return_sized_n 'int * __sized_by_or_null(size)(int)'

// CHECK: asdf_return_ended 'int * __ended_by(end)(int *)'

// CHECK: asdf_return_sized_mul 'int * __sized_by(size * count)(int, int)'


// CHECK: asdf_return_counted_out 'int * __counted_by(*len)(int *)'

// CHECK: asdf_return_counted_const 'int * __counted_by(7)(int *)'

// CHECK: asdf_return_counted_nullable 'int * __counted_by(len) _Nullable (int)'

// CHECK: asdf_return_counted_default_level 'int * __counted_by(len)(int)'

// CHECK: asdf_return_counted_redundant 'int * __counted_by(len)(int)'

// CHECK: asdf_return_ended_chained 'int * __ended_by(mid)(int * __ended_by(end), int * /* __started_by(mid) */ )'
// CHECK: 'int * __ended_by(end)':'int *'
// CHECK: end 'int * /* __started_by(mid) */ ':'int *'

// CHECK: asdf_return_ended_redundant 'int * __ended_by(end)(int *)'
// CHECK: end 'int *'

// CHECK: asdf_return_nterm 'char * __terminated_by(0)()'

