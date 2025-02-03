// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fsyntax-only -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks -fexperimental-bounds-safety-attributes %s -ast-dump -ast-dump-filter asdf | FileCheck %s

#include "BoundsUnsafeObjC.h"

// CHECK-LABEL: asdf_counted
// CHECK: buf 'int * __counted_by(len)':'int *'
// CHECK-NEXT: len 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0

// CHECK-LABEL: asdf_sized
// CHECK: buf 'int * __sized_by(size)':'int *'
// CHECK-NEXT: size 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0

// CHECK-LABEL: asdf_counted_n
// CHECK: buf 'int * __counted_by_or_null(len)':'int *'
// CHECK-NEXT: len 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0

// CHECK-LABEL: asdf_sized_n
// CHECK: buf 'int * __sized_by_or_null(size)':'int *'
// CHECK-NEXT: size 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0

// CHECK-LABEL: asdf_ended
// CHECK: buf 'int * __ended_by(end)':'int *'
// CHECK: end 'int * /* __started_by(buf) */ ':'int *'

// CHECK-LABEL: asdf_sized_mul
// CHECK: buf 'int * __sized_by(size * count)':'int *'
// CHECK-NEXT: size 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0
// CHECK-NEXT: count 'int'
// CHECK-NEXT: DependerDeclsAttr {{.*}} 0

// CHECK-LABEL: asdf_counted_out
// CHECK: buf 'int * __counted_by(*len)*'
// CHECK-NEXT: len 'int *'
// CHECK-NEXT: DependerDeclsAttr {{.*}} IsDeref {{.*}} 1

// CHECK-LABEL: asdf_counted_const
// CHECK: buf 'int * __counted_by(7)':'int *'

// CHECK-LABEL: asdf_counted_nullable
// CHECK: buf 'int * __counted_by(len) _Nullable':'int *'

// CHECK-LABEL: asdf_counted_noescape
// CHECK: buf 'int * __counted_by(len)':'int *'
// CHECK-NEXT: NoEscapeAttr

// CHECK-LABEL: asdf_nterm
// CHECK: buf 'int * __terminated_by(0)':'int *'

