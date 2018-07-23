// This test requires linux because it uses `diff` and compares filepaths
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --public --extra-arg=-fmodules-ts --doxygen -p %t %t/test.cpp -output=%t/docs-with-public-flag
// RUN: clang-doc --extra-arg=-fmodules-ts --doxygen -p %t %t/test.cpp -output=%t/docs-without
// RUN: cat %t/docs-with-public-flag/moduleFunction.yaml | FileCheck %s --check-prefix=CHECK-A
// RUN: cat %t/docs-with-public-flag/exportedModuleFunction.yaml | FileCheck %s --check-prefix=CHECK-B
// RUN: (diff -qry %t/docs-with-public-flag %t/docs-without | sed 's:.*/::' > %t/public.diff) || true
// RUN: cat %t/public.diff | FileCheck %s --check-prefix=CHECK-C

export module M;

int moduleFunction(int x); //ModuleLinkage
// CHECK-A: ---
// CHECK-A-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-A-NEXT: Name:            'moduleFunction'
// CHECK-A-NEXT: Location:
// CHECK-A-NEXT:   - LineNumber:      16
// CHECK-A-NEXT:     Filename:        {{.*}}
// CHECK-A-NEXT: Params:
// CHECK-A-NEXT:   - Type:
// CHECK-A-NEXT:       Name:            'int'
// CHECK-A-NEXT:     Name:            'x'
// CHECK-A-NEXT: ReturnType:
// CHECK-A-NEXT:   Type:
// CHECK-A-NEXT:     Name:            'int'
// CHECK-A-NEXT: ...

static int staticModuleFunction(int x); //ModuleInternalLinkage

export double exportedModuleFunction(double y, int z); //ExternalLinkage
// CHECK-B: ---
// CHECK-B-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-B-NEXT: Name:            'exportedModuleFunction'
// CHECK-B-NEXT: Location:
// CHECK-B-NEXT:   - LineNumber:      34
// CHECK-B-NEXT:     Filename:        {{.*}}
// CHECK-B-NEXT: Params:
// CHECK-B-NEXT:   - Type:
// CHECK-B-NEXT:       Name:            'double'
// CHECK-B-NEXT:     Name:            'y'
// CHECK-B-NEXT:   - Type:
// CHECK-B-NEXT:       Name:            'int'
// CHECK-B-NEXT:     Name:            'z'
// CHECK-B-NEXT: ReturnType:
// CHECK-B-NEXT:   Type:
// CHECK-B-NEXT:     Name:            'double'
// CHECK-B-NEXT: ...

// CHECK-C: docs-without: staticModuleFunction.yaml
