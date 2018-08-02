// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --extra-arg=-fmodules-ts --doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/moduleFunction.yaml | FileCheck %s --check-prefix=CHECK-A
// RUN: cat %t/docs/staticModuleFunction.yaml | FileCheck %s --check-prefix=CHECK-B
// RUN: cat %t/docs/exportedModuleFunction.yaml | FileCheck %s --check-prefix=CHECK-C

export module M;

int moduleFunction(int x); //ModuleLinkage
// CHECK-A: ---
// CHECK-A-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-A-NEXT: Name:            'moduleFunction'
// CHECK-A-NEXT: Location:
// CHECK-A-NEXT:   - LineNumber:      12
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
// CHECK-B: ---
// CHECK-B-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-B-NEXT: Name:            'staticModuleFunction'
// CHECK-B-NEXT: Location:
// CHECK-B-NEXT:   - LineNumber:      28
// CHECK-B-NEXT:     Filename:        {{.*}}
// CHECK-B-NEXT: Params:
// CHECK-B-NEXT:   - Type:
// CHECK-B-NEXT:       Name:            'int'
// CHECK-B-NEXT:     Name:            'x'
// CHECK-B-NEXT: ReturnType:
// CHECK-B-NEXT:   Type:
// CHECK-B-NEXT:     Name:            'int'
// CHECK-B-NEXT: ...

export double exportedModuleFunction(double y, int z); //ExternalLinkage
// CHECK-C: ---
// CHECK-C-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-C-NEXT: Name:            'exportedModuleFunction'
// CHECK-C-NEXT: Location:
// CHECK-C-NEXT:   - LineNumber:      44
// CHECK-C-NEXT:     Filename:        {{.*}}
// CHECK-C-NEXT: Params:
// CHECK-C-NEXT:   - Type:
// CHECK-C-NEXT:       Name:            'double'
// CHECK-C-NEXT:     Name:            'y'
// CHECK-C-NEXT:   - Type:
// CHECK-C-NEXT:       Name:            'int'
// CHECK-C-NEXT:     Name:            'z'
// CHECK-C-NEXT: ReturnType:
// CHECK-C-NEXT:   Type:
// CHECK-C-NEXT:     Name:            'double'
// CHECK-C-NEXT: ...
