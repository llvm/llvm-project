// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --output=%t --executor=standalone %S/../Inputs/namespace-html-md.cpp
// RUN: FileCheck %s < %t/md/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.md -check-prefix=MD-ANON-CLASS-LINE
// RUN: FileCheck %s < %t/md/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.md -check-prefix=MD-ANON-CLASS
// RUN: FileCheck %s < %t/md/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX-LINE
// RUN: FileCheck %s < %t/md/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX
// RUN: FileCheck %s < %t/md/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.md -check-prefix=MD-ANOTHER-CLASS-LINE
// RUN: FileCheck %s < %t/md/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.md -check-prefix=MD-ANOTHER-CLASS
// RUN: FileCheck %s < %t/md/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX-LINE
// RUN: FileCheck %s < %t/md/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.md -check-prefix=MD-NESTED-CLASS-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.md -check-prefix=MD-NESTED-CLASS
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.md -check-prefix=MD-PRIMARY-CLASS-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.md -check-prefix=MD-PRIMARY-CLASS
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md -check-prefix=MD-GLOBAL-INDEX
// RUN: FileCheck %s < %t/md/all_files.md -check-prefix=MD-ALL-FILES
// RUN: FileCheck %s < %t/md/index.md -check-prefix=MD-INDEX

// MD-ANON-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#3*

// MD-ANON-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#4*

// MD-ANON-CLASS: # class AnonClass

// MD-ANON-INDEX: # namespace @nonymous_namespace
// MD-ANON-INDEX: Anonymous Namespace
// MD-ANON-INDEX: ## Records
// MD-ANON-INDEX: * [AnonClass](AnonClass.md)
// MD-ANON-INDEX: ## Functions
// MD-ANON-INDEX: ### anonFunction
// MD-ANON-INDEX: *void anonFunction()*

// MD-PRIMARY-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#10*

// MD-PRIMARY-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#12*

// MD-PRIMARY-CLASS: # class ClassInPrimaryNamespace
// MD-PRIMARY-CLASS: Class in PrimaryNamespace

// MD-NESTED-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#17*

// MD-NESTED-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#19*

// MD-NESTED-CLASS: # class ClassInNestedNamespace
// MD-NESTED-CLASS: Class in NestedNamespace

// MD-NESTED-INDEX: # namespace NestedNamespace
// MD-NESTED-INDEX: Nested namespace
// MD-NESTED-INDEX: ## Records
// MD-NESTED-INDEX: * [ClassInNestedNamespace](ClassInNestedNamespace.md)
// MD-NESTED-INDEX: ## Functions
// MD-NESTED-INDEX: ### functionInNestedNamespace
// MD-NESTED-INDEX: *void functionInNestedNamespace()*
// MD-NESTED-INDEX: Function in NestedNamespace

// MD-PRIMARY-INDEX: # namespace PrimaryNamespace
// MD-PRIMARY-INDEX: Primary Namespace
// MD-PRIMARY-INDEX: ## Namespaces
// MD-PRIMARY-INDEX: * [NestedNamespace](NestedNamespace{{[\/]}}index.md)
// MD-PRIMARY-INDEX: ## Records
// MD-PRIMARY-INDEX: * [ClassInPrimaryNamespace](ClassInPrimaryNamespace.md)
// MD-PRIMARY-INDEX: ## Functions
// MD-PRIMARY-INDEX: ### functionInPrimaryNamespace
// MD-PRIMARY-INDEX: *void functionInPrimaryNamespace()*
// MD-PRIMARY-INDEX: Function in PrimaryNamespace

// MD-ANOTHER-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#26*

// MD-ANOTHER-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#28*

// MD-ANOTHER-CLASS: # class ClassInAnotherNamespace
// MD-ANOTHER-CLASS: Class in AnotherNamespace

// MD-ANOTHER-INDEX: # namespace AnotherNamespace
// MD-ANOTHER-INDEX: AnotherNamespace
// MD-ANOTHER-INDEX: ## Records
// MD-ANOTHER-INDEX: * [ClassInAnotherNamespace](ClassInAnotherNamespace.md)
// MD-ANOTHER-INDEX: ## Functions
// MD-ANOTHER-INDEX: ### functionInAnotherNamespace
// MD-ANOTHER-INDEX: *void functionInAnotherNamespace()*
// MD-ANOTHER-INDEX: Function in AnotherNamespace

// MD-GLOBAL-INDEX: # Global Namespace
// MD-GLOBAL-INDEX: ## Namespaces
// MD-GLOBAL-INDEX: * [@nonymous_namespace]({{.*}}{{[\/]}}@nonymous_namespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [AnotherNamespace]({{.*}}{{[\/]}}AnotherNamespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [PrimaryNamespace]({{.*}}{{[\/]}}PrimaryNamespace{{[\/]}}index.md)

// MD-ALL-FILES: # All Files
// MD-ALL-FILES: ## [@nonymous_namespace](@nonymous_namespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [AnotherNamespace](AnotherNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [GlobalNamespace](GlobalNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [PrimaryNamespace](PrimaryNamespace{{[\/]}}index.md)

// MD-INDEX: #  C/C++ Reference
// MD-INDEX: * Namespace: [@nonymous_namespace](@nonymous_namespace)
// MD-INDEX: * Namespace: [AnotherNamespace](AnotherNamespace)
// MD-INDEX: * Namespace: [PrimaryNamespace](PrimaryNamespace)
