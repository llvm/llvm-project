// RUN: clang-tblgen -gen-clang-attr-impl -I%p/../../include \
// RUN:   %p/../../include/clang/Basic/Attr.td -o - | FileCheck %s

// CHECK: struct AttrPrintSpellingInfo {
// CHECK: static_assert(sizeof(AttrPrintSpellingInfo) == 6);

// CHECK-LABEL: void AlignedAttr::printPretty(
// CHECK: llvm::StringRef Suffix = printAttributeSpelling(
// CHECK-NEXT: OS, {{[0-9]+}}, 6, getAttributeSpellingListIndex());
// CHECK: alignmentExpr->printPretty(OS, nullptr, Policy);
// CHECK-NOT: alignmentExpr->printPretty(OS, nullptr, Policy);
// CHECK: OS << Suffix;
// CHECK-NEXT: }

// CHECK-LABEL: void AvailabilityAttr::printPretty(
// CHECK: switch (getAttributeSpellingListIndex()) {
// CHECK: case 0
// CHECK: OS << "__attribute__((availability";
// CHECK: case 1
// CHECK: OS << "{{\[\[}}clang::availability";

// CHECK-LABEL: void LoopHintAttr::printPretty(
// CHECK: llvm::StringRef Suffix = printAttributeSpelling(
// CHECK-NEXT: OS, {{[0-9]+}}, 5, getAttributeSpellingListIndex());
// CHECK-NEXT: printPrettyPragma(OS, Policy);
// CHECK-NEXT: OS << Suffix;
