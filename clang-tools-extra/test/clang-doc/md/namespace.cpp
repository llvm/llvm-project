// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --output=%t --executor=standalone %S/../Inputs/namespace-html-md.cpp
// RUN: FileCheck %s < %t/@nonymous_namespace/AnonClass.md -check-prefix=MD-ANON-CLASS-LINE
// RUN: FileCheck %s < %t/@nonymous_namespace/AnonClass.md -check-prefix=MD-ANON-CLASS
// RUN: FileCheck %s < %t/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX-LINE
// RUN: FileCheck %s < %t/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX
// RUN: FileCheck %s < %t/AnotherNamespace/ClassInAnotherNamespace.md -check-prefix=MD-ANOTHER-CLASS-LINE
// RUN: FileCheck %s < %t/AnotherNamespace/ClassInAnotherNamespace.md -check-prefix=MD-ANOTHER-CLASS
// RUN: FileCheck %s < %t/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX-LINE
// RUN: FileCheck %s < %t/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.md -check-prefix=MD-NESTED-CLASS-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.md -check-prefix=MD-NESTED-CLASS
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/ClassInPrimaryNamespace.md -check-prefix=MD-PRIMARY-CLASS-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/ClassInPrimaryNamespace.md -check-prefix=MD-PRIMARY-CLASS
// RUN: FileCheck %s < %t/GlobalNamespace/index.md -check-prefix=MD-GLOBAL-INDEX
// RUN: FileCheck %s < %t/all_files.md -check-prefix=MD-ALL-FILES
// RUN: FileCheck %s < %t/index.md -check-prefix=MD-INDEX
// RUN: clang-doc --format=md_mustache --output=%t --executor=standalone %S/../Inputs/namespace-html-md.cpp
// RUN: FileCheck %s < %t/md/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.md -check-prefix=MD-MUSTACHE-ANON-CLASS-LINE
// RUN: FileCheck %s < %t/md/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.md -check-prefix=MD-MUSTACHE-ANON-CLASS
// RUN: FileCheck %s < %t/md/@nonymous_namespace/index.md -check-prefix=MD-MUSTACHE-ANON-INDEX-LINE
// RUN: FileCheck %s < %t/md/@nonymous_namespace/index.md -check-prefix=MD-MUSTACHE-ANON-INDEX
// RUN: FileCheck %s < %t/md/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.md -check-prefix=MD-MUSTACHE-ANOTHER-CLASS-LINE
// RUN: FileCheck %s < %t/md/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.md -check-prefix=MD-MUSTACHE-ANOTHER-CLASS
// RUN: FileCheck %s < %t/md/AnotherNamespace/index.md -check-prefix=MD-MUSTACHE-ANOTHER-INDEX-LINE
// RUN: FileCheck %s < %t/md/AnotherNamespace/index.md -check-prefix=MD-MUSTACHE-ANOTHER-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.md -check-prefix=MD-MUSTACHE-NESTED-CLASS-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.md -check-prefix=MD-MUSTACHE-NESTED-CLASS
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-MUSTACHE-NESTED-INDEX-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-MUSTACHE-NESTED-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/index.md -check-prefix=MD-MUSTACHE-PRIMARY-INDEX-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/index.md -check-prefix=MD-MUSTACHE-PRIMARY-INDEX
// RUN: FileCheck %s < %t/md/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.md -check-prefix=MD-MUSTACHE-PRIMARY-CLASS-LINE
// RUN: FileCheck %s < %t/md/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.md -check-prefix=MD-MUSTACHE-PRIMARY-CLASS
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md -check-prefix=MD-MUSTACHE-GLOBAL-INDEX
// RUN: FileCheck %s < %t/md/all_files.md -check-prefix=MD-MUSTACHE-ALL-FILES
// RUN: FileCheck %s < %t/md/index.md -check-prefix=MD-MUSTACHE-INDEX

// MD-ANON-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#3*
// MD-MUSTACHE-ANON-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#3*

// MD-ANON-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#4*
// MD-MUSTACHE-ANON-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#4*

// MD-ANON-CLASS: # class AnonClass
// MD-MUSTACHE-ANON-CLASS: # class AnonClass

// MD-ANON-INDEX: # namespace @nonymous_namespace
// MD-ANON-INDEX:  Anonymous Namespace
// MD-ANON-INDEX: ## Records
// MD-ANON-INDEX: * [AnonClass](AnonClass.md)
// MD-ANON-INDEX: ## Functions
// MD-ANON-INDEX: ### anonFunction
// MD-ANON-INDEX: *void anonFunction()*

// MD-MUSTACHE-ANON-INDEX: # namespace @nonymous_namespace
// MD-MUSTACHE-ANON-INDEX: Anonymous Namespace
// MD-MUSTACHE-ANON-INDEX: ## Records
// MD-MUSTACHE-ANON-INDEX: * [AnonClass](AnonClass.md)
// MD-MUSTACHE-ANON-INDEX: ## Functions
// MD-MUSTACHE-ANON-INDEX: ### anonFunction
// MD-MUSTACHE-ANON-INDEX: *void anonFunction()*

// MD-PRIMARY-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#10*
// MD-MUSTACHE-PRIMARY-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#10*

// MD-PRIMARY-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#12*
// MD-MUSTACHE-PRIMARY-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#12*

// MD-PRIMARY-CLASS: # class ClassInPrimaryNamespace
// MD-PRIMARY-CLASS: Class in PrimaryNamespace

// MD-MUSTACHE-PRIMARY-CLASS: # class ClassInPrimaryNamespace
// MD-MUSTACHE-PRIMARY-CLASS: Class in PrimaryNamespace

// MD-NESTED-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#17*
// MD-MUSTACHE-NESTED-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#17*

// MD-NESTED-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#19*
// MD-MUSTACHE-NESTED-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#19*

// MD-NESTED-CLASS: # class ClassInNestedNamespace
// MD-NESTED-CLASS: Class in NestedNamespace

// MD-MUSTACHE-NESTED-CLASS: # class ClassInNestedNamespace
// MD-MUSTACHE-NESTED-CLASS: Class in NestedNamespace

// MD-NESTED-INDEX: # namespace NestedNamespace
// MD-NESTED-INDEX: Nested namespace
// MD-NESTED-INDEX: ## Records
// MD-NESTED-INDEX: * [ClassInNestedNamespace](ClassInNestedNamespace.md)
// MD-NESTED-INDEX: ## Functions
// MD-NESTED-INDEX: ### functionInNestedNamespace
// MD-NESTED-INDEX: *void functionInNestedNamespace()*
// MD-NESTED-INDEX: Function in NestedNamespace

// MD-MUSTACHE-NESTED-INDEX: # namespace NestedNamespace
// MD-MUSTACHE-NESTED-INDEX: Nested namespace
// MD-MUSTACHE-NESTED-INDEX: ## Records
// MD-MUSTACHE-NESTED-INDEX: * [ClassInNestedNamespace](ClassInNestedNamespace.md)
// MD-MUSTACHE-NESTED-INDEX: ## Functions
// MD-MUSTACHE-NESTED-INDEX: ### functionInNestedNamespace
// MD-MUSTACHE-NESTED-INDEX: *void functionInNestedNamespace()*
// MD-MUSTACHE-NESTED-INDEX: Function in NestedNamespace

// MD-PRIMARY-INDEX: # namespace PrimaryNamespace
// MD-PRIMARY-INDEX:  Primary Namespace
// MD-PRIMARY-INDEX: ## Namespaces
// MD-PRIMARY-INDEX: * [NestedNamespace](NestedNamespace{{[\/]}}index.md)
// MD-PRIMARY-INDEX: ## Records
// MD-PRIMARY-INDEX: * [ClassInPrimaryNamespace](ClassInPrimaryNamespace.md)
// MD-PRIMARY-INDEX: ## Functions
// MD-PRIMARY-INDEX: ### functionInPrimaryNamespace
// MD-PRIMARY-INDEX: *void functionInPrimaryNamespace()*
// MD-PRIMARY-INDEX:  Function in PrimaryNamespace

// MD-MUSTACHE-PRIMARY-INDEX: # namespace PrimaryNamespace
// MD-MUSTACHE-PRIMARY-INDEX: Primary Namespace
// MD-MUSTACHE-PRIMARY-INDEX: ## Namespaces
// MD-MUSTACHE-PRIMARY-INDEX: * [NestedNamespace](NestedNamespace{{[\/]}}index.md)
// MD-MUSTACHE-PRIMARY-INDEX: ## Records
// MD-MUSTACHE-PRIMARY-INDEX: * [ClassInPrimaryNamespace](ClassInPrimaryNamespace.md)
// MD-MUSTACHE-PRIMARY-INDEX: ## Functions
// MD-MUSTACHE-PRIMARY-INDEX: ### functionInPrimaryNamespace
// MD-MUSTACHE-PRIMARY-INDEX: *void functionInPrimaryNamespace()*
// MD-MUSTACHE-PRIMARY-INDEX: Function in PrimaryNamespace

// MD-ANOTHER-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#26*
// MD-MUSTACHE-ANOTHER-INDEX-LINE: *Defined at {{.*}}namespace-html-md.cpp#26*

// MD-ANOTHER-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#28*
// MD-MUSTACHE-ANOTHER-CLASS-LINE: *Defined at {{.*}}namespace-html-md.cpp#28*

// MD-ANOTHER-CLASS: # class ClassInAnotherNamespace
// MD-ANOTHER-CLASS:  Class in AnotherNamespace

// MD-MUSTACHE-ANOTHER-CLASS: # class ClassInAnotherNamespace
// MD-MUSTACHE-ANOTHER-CLASS: Class in AnotherNamespace

// MD-ANOTHER-INDEX: # namespace AnotherNamespace
// MD-ANOTHER-INDEX: AnotherNamespace
// MD-ANOTHER-INDEX: ## Records
// MD-ANOTHER-INDEX: * [ClassInAnotherNamespace](ClassInAnotherNamespace.md)
// MD-ANOTHER-INDEX: ## Functions
// MD-ANOTHER-INDEX: ### functionInAnotherNamespace
// MD-ANOTHER-INDEX: *void functionInAnotherNamespace()*
// MD-ANOTHER-INDEX: Function in AnotherNamespace

// MD-MUSTACHE-ANOTHER-INDEX: # namespace AnotherNamespace
// MD-MUSTACHE-ANOTHER-INDEX: AnotherNamespace
// MD-MUSTACHE-ANOTHER-INDEX: ## Records
// MD-MUSTACHE-ANOTHER-INDEX: * [ClassInAnotherNamespace](ClassInAnotherNamespace.md)
// MD-MUSTACHE-ANOTHER-INDEX: ## Functions
// MD-MUSTACHE-ANOTHER-INDEX: ### functionInAnotherNamespace
// MD-MUSTACHE-ANOTHER-INDEX: *void functionInAnotherNamespace()*
// MD-MUSTACHE-ANOTHER-INDEX: Function in AnotherNamespace

// MD-GLOBAL-INDEX: # Global Namespace
// MD-GLOBAL-INDEX: ## Namespaces
// MD-GLOBAL-INDEX: * [@nonymous_namespace](..{{[\/]}}@nonymous_namespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [AnotherNamespace](..{{[\/]}}AnotherNamespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [PrimaryNamespace](..{{[\/]}}PrimaryNamespace{{[\/]}}index.md)

// MD-MUSTACHE-GLOBAL-INDEX: # Global Namespace
// MD-MUSTACHE-GLOBAL-INDEX: ## Namespaces
// MD-MUSTACHE-GLOBAL-INDEX: * [@nonymous_namespace]({{.*}}{{[\/]}}@nonymous_namespace{{[\/]}}index.md)
// MD-MUSTACHE-GLOBAL-INDEX: * [AnotherNamespace]({{.*}}{{[\/]}}AnotherNamespace{{[\/]}}index.md)
// MD-MUSTACHE-GLOBAL-INDEX: * [PrimaryNamespace]({{.*}}{{[\/]}}PrimaryNamespace{{[\/]}}index.md)

// MD-ALL-FILES: # All Files
// MD-ALL-FILES: ## [@nonymous_namespace](@nonymous_namespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [AnotherNamespace](AnotherNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [GlobalNamespace](GlobalNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [PrimaryNamespace](PrimaryNamespace{{[\/]}}index.md)

// MD-MUSTACHE-ALL-FILES: # All Files
// MD-MUSTACHE-ALL-FILES: ## [@nonymous_namespace](@nonymous_namespace{{[\/]}}index.md)
// MD-MUSTACHE-ALL-FILES: ## [AnotherNamespace](AnotherNamespace{{[\/]}}index.md)
// MD-MUSTACHE-ALL-FILES: ## [GlobalNamespace](GlobalNamespace{{[\/]}}index.md)
// MD-MUSTACHE-ALL-FILES: ## [PrimaryNamespace](PrimaryNamespace{{[\/]}}index.md)

// MD-INDEX: #  C/C++ Reference
// MD-INDEX: * Namespace: [@nonymous_namespace](@nonymous_namespace)
// MD-INDEX: * Namespace: [AnotherNamespace](AnotherNamespace)
// MD-INDEX: * Namespace: [PrimaryNamespace](PrimaryNamespace)

// MD-MUSTACHE-INDEX: #  C/C++ Reference
// MD-MUSTACHE-INDEX: * Namespace: [@nonymous_namespace](@nonymous_namespace)
// MD-MUSTACHE-INDEX: * Namespace: [AnotherNamespace](AnotherNamespace)
// MD-MUSTACHE-INDEX: * Namespace: [PrimaryNamespace](PrimaryNamespace)
