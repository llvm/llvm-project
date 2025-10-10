#include <ptrcheck.h>
//==============================================================================
// Detailed trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fbounds-safety -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=ANNOTATE,DETAILED

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fbounds-safety \
// RUN: -fbounds-safety-debug-trap-reasons=detailed -emit-llvm %s -o - | FileCheck %s --check-prefixes=ANNOTATE,DETAILED

//==============================================================================
// Basic trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fbounds-safety \
// RUN: -fbounds-safety-debug-trap-reasons=basic -emit-llvm %s -o - | FileCheck %s --check-prefixes=ANNOTATE,BASIC

//==============================================================================
// No trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fbounds-safety \
// RUN: -fbounds-safety-debug-trap-reasons=none -emit-llvm %s -o - | FileCheck %s --check-prefix=NO-ANNOTATE

int read(int* __bidi_indexable ptr, int idx) { return ptr[idx]; }

// ANNOTATE-LABEL: @read
// ANNOTATE: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[LOC:![0-9]+]]
// ANNOTATE: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// DETAILED: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$indexing above upper bound in 'ptr[idx]'"
// BASIC: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Dereferencing above bounds"

// NO-ANNOTATE-LABEL: @read
// NO-ANNOTATE: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[LOC:![0-9]+]]
// NO-ANNOTATE-NOT: __clang_trap_msg
