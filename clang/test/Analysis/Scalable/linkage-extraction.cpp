// Verifies the linkage the extractor records for each source construct.
//
// Binding and coalescing are independent axes: an inline function binds
// strongly and additionally guarantees its definitions are identical, whereas
// __attribute__((weak)) binds weakly with no such guarantee. Object formats
// encode this differently — ELF and Mach-O lower an ODR definition to a weak
// symbol while COFF keeps it strong and uses a COMDAT — so the summary records
// the source-level fact and leaves the lowering to the linker. See
// docs/ssaf-linker-elf-behavior.md §3 and docs/ssaf-linker-coff-behavior.md §2.
//
// Each construct lives in its own file so that the summary holds exactly one
// entity: ids are assigned in traversal order, so a shared file would make the
// checks depend on declaration order rather than on the linkage itself.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// An ordinary definition: strong, with no coalescing guarantee.
// RUN: %clang_cc1 -fsyntax-only %t/ordinary.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id=cu \
// RUN:   --ssaf-tu-summary-file=%t/ordinary.json
// RUN: cat %t/ordinary.json | FileCheck %s --check-prefix=ORDINARY
// ORDINARY:      "usr": "c:@F@ordinary#"
// ORDINARY:      "binding": "Strong",
// ORDINARY-NEXT: "coalescing": "None",
// ORDINARY-NEXT: "definition": "Definition",

// An inline definition: still strong, but its copies must be identical.
// RUN: %clang_cc1 -fsyntax-only %t/inlined.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id=cu \
// RUN:   --ssaf-tu-summary-file=%t/inlined.json
// RUN: cat %t/inlined.json | FileCheck %s --check-prefix=INLINED
// INLINED:      "usr": "c:@F@inlined#"
// INLINED:      "binding": "Strong",
// INLINED-NEXT: "coalescing": "ODR",
// INLINED-NEXT: "definition": "Definition",

// An explicitly weak definition may be replaced by an unrelated one, so it
// carries no ODR guarantee even though it is also inline.
// RUN: %clang_cc1 -fsyntax-only %t/weak_inlined.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id=cu \
// RUN:   --ssaf-tu-summary-file=%t/weak_inlined.json
// RUN: cat %t/weak_inlined.json | FileCheck %s --check-prefix=WEAK
// WEAK:      "usr": "c:@F@weak_inlined#"
// WEAK:      "binding": "Weak",
// WEAK-NEXT: "coalescing": "None",
// WEAK-NEXT: "definition": "Definition",

// A hidden definition keeps its visibility; the extractor records what the
// source said and leaves target-specific coercion to the linker.
// RUN: %clang_cc1 -fsyntax-only %t/hidden.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id=cu \
// RUN:   --ssaf-tu-summary-file=%t/hidden.json
// RUN: cat %t/hidden.json | FileCheck %s --check-prefix=HIDDEN
// HIDDEN:      "usr": "c:@F@hidden_fn#"
// HIDDEN:      "binding": "Strong",
// HIDDEN-NEXT: "coalescing": "None",
// HIDDEN-NEXT: "definition": "Definition",
// HIDDEN-NEXT: "type": "External",
// HIDDEN-NEXT: "visibility": "Hidden"

//--- ordinary.cpp
int ordinary(void) { return 1; }

//--- inlined.cpp
inline int inlined(void) { return 2; }

//--- weak_inlined.cpp
__attribute__((weak)) inline int weak_inlined(void) { return 3; }

//--- hidden.cpp
__attribute__((visibility("hidden"))) int hidden_fn(void) { return 4; }
