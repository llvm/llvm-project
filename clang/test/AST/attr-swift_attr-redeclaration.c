// RUN: %clang_cc1 -ast-dump -ast-dump-filter=two_on_first %s \
// RUN:   | FileCheck --check-prefix=TWO --implicit-check-not=SwiftAttrAttr %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter=split %s \
// RUN:   | FileCheck --check-prefix=SPLIT --implicit-check-not=SwiftAttrAttr %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter=dedup %s \
// RUN:   | FileCheck --check-prefix=DEDUP --implicit-check-not=SwiftAttrAttr %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter=overlap %s \
// RUN:   | FileCheck --check-prefix=OVERLAP --implicit-check-not=SwiftAttrAttr %s

// 'swift_attr' is a bag of independent annotations identified by its string
// argument, so a redeclaration inherits all of them, not just the first.

__attribute__((swift_attr("@a"))) __attribute__((swift_attr("@b")))
void two_on_first(void);
void two_on_first(void);

// TWO:      FunctionDecl {{.*}} two_on_first
// TWO-NEXT:   SwiftAttrAttr {{.*}} "@a"
// TWO-NEXT:   SwiftAttrAttr {{.*}} "@b"
// TWO:      FunctionDecl {{.*}} prev {{.*}} two_on_first
// TWO-NEXT:   SwiftAttrAttr {{.*}} Inherited "@a"
// TWO-NEXT:   SwiftAttrAttr {{.*}} Inherited "@b"

// A 'swift_attr' on the redeclaration must not suppress inheriting the
// different ones from the previous declaration.
__attribute__((swift_attr("@x1"))) __attribute__((swift_attr("@x2")))
void split(void);
__attribute__((swift_attr("@y1"))) __attribute__((swift_attr("@y2")))
void split(void);

// SPLIT:      FunctionDecl {{.*}} split
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} "@x1"
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} "@x2"
// SPLIT:      FunctionDecl {{.*}} prev {{.*}} split
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} Inherited "@x1"
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} Inherited "@x2"
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} "@y1"
// SPLIT-NEXT:   SwiftAttrAttr {{.*}} "@y2"

// Identical annotations are still deduplicated, so a chain of redeclarations
// does not accumulate copies.
__attribute__((swift_attr("@same1"))) __attribute__((swift_attr("@same2")))
void dedup(void);
__attribute__((swift_attr("@same1"))) __attribute__((swift_attr("@same2")))
void dedup(void);
void dedup(void);

// DEDUP:      FunctionDecl {{.*}} dedup
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} "@same1"
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} "@same2"
// DEDUP:      FunctionDecl {{.*}} prev {{.*}} dedup
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} "@same1"
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} "@same2"
// DEDUP:      FunctionDecl {{.*}} prev {{.*}} dedup
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} Inherited "@same1"
// DEDUP-NEXT:   SwiftAttrAttr {{.*}} Inherited "@same2"

// Partially overlapping sets are merged into their union.
__attribute__((swift_attr("@p"))) __attribute__((swift_attr("@q")))
void overlap(void);
__attribute__((swift_attr("@q"))) __attribute__((swift_attr("@r")))
void overlap(void);

// OVERLAP:      FunctionDecl {{.*}} overlap
// OVERLAP-NEXT:   SwiftAttrAttr {{.*}} "@p"
// OVERLAP-NEXT:   SwiftAttrAttr {{.*}} "@q"
// OVERLAP:      FunctionDecl {{.*}} prev {{.*}} overlap
// OVERLAP-NEXT:   SwiftAttrAttr {{.*}} Inherited "@p"
// OVERLAP-NEXT:   SwiftAttrAttr {{.*}} "@q"
// OVERLAP-NEXT:   SwiftAttrAttr {{.*}} "@r"
