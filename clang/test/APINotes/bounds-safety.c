// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter asdf | FileCheck %s

// FIXME: Bounds safety annotations are parsed and stored in the APINotes
// format, but are not yet applied as attributes in the AST upstream since
// __counted_by and related attributes are not yet supported in function
// signatures. Once that support is added, update this test to verify the
// annotations appear on the relevant declarations.

#include "BoundsUnsafe.h"

// CHECK: imported in BoundsUnsafe asdf_counted 'void (int *, int)'
// CHECK: imported in BoundsUnsafe buf 'int *'

// CHECK: imported in BoundsUnsafe asdf_sized 'void (void *, int)'
// CHECK: imported in BoundsUnsafe buf 'void *'

// CHECK: imported in BoundsUnsafe asdf_counted_n 'void (int *, int)'
// CHECK: imported in BoundsUnsafe buf 'int *'

// CHECK: imported in BoundsUnsafe asdf_sized_n 'void (void *, int)'
// CHECK: imported in BoundsUnsafe buf 'void *'

// CHECK: imported in BoundsUnsafe asdf_ended 'void (int *, int *)'
// CHECK: imported in BoundsUnsafe buf 'int *'

// CHECK: imported in BoundsUnsafe asdf_counted_indirect 'void (int **, int)'
// CHECK: imported in BoundsUnsafe buf 'int **'
