// Tests clang-ssaf-src-edit-merge against real multi-TU source.
// Specifically, this file tests non-conflicting merges across multiple TUs.

// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// DEFINE: %{casedir} = unset
// DEFINE: %{testname} = unset
// DEFINE: %{tu_json} = %{casedir}/%{testname}.tu.json
// DEFINE: %{lu_json} = %{casedir}/%{testname}.lu.json
// DEFINE: %{wpa_json} = %{casedir}/%{testname}.wpa.json
// DEFINE: %{apply_dir} = %{casedir}/%{testname}_apply
// DEFINE: %{edits_yaml} = %{apply_dir}/%{testname}.edits.yaml
// DEFINE: %{extract} = %clang -fsyntax-only %{casedir}/%{testname}.cpp \
// DEFINE:   --ssaf-extract-summaries=PointerFlow,UnsafeBufferUsage \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu --ssaf-tu-summary-file=%{tu_json}
// DEFINE: %{link} = clang-ssaf-linker %{tu_json} -o %{lu_json}
// DEFINE: %{analyze} = clang-ssaf-analyzer %{lu_json} -o %{wpa_json} -a UnsafeBufferReachableAnalysisResult
// DEFINE: %{transform} = mkdir -p %{apply_dir} && %clang -fsyntax-only %{casedir}/%{testname}.cpp \
// DEFINE:   --ssaf-source-transformation=cpp-bounded-buffers \
// DEFINE:   --ssaf-global-scope-analysis-result=%{wpa_json} \
// DEFINE:   --ssaf-src-edit-file=%{edits_yaml} \
// DEFINE:   --ssaf-transformation-report-file=%{casedir}/%{testname}.report.sarif \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu --ssaf-link-unit-id=%{testname}.lu
// DEFINE: %{pipeline} = %{extract} && %{link} && %{analyze} && %{transform}
// DEFINE: %{merge_inputs} = unset
// DEFINE: %{merge} = mkdir -p %{casedir}/merge_apply && clang-ssaf-src-edit-merge %{merge_inputs} -o %{casedir}/merge_apply/merged.yaml --sarif-conflicts-out=%{casedir}/conflicts.sarif 2> %{casedir}/merge.stderr
// DEFINE: %{apply} = clang-apply-replacements %{casedir}/merge_apply

//--- distinct-files/a.h
int *ga;

//--- distinct-files/b.h
int *gb;

//--- distinct-files/a.cpp
#include "a.h"
void use_a() { ga[5] = 0; }

//--- distinct-files/b.cpp
#include "b.h"
void use_b() { gb[7] = 0; }

// REDEFINE: %{casedir} = %t/distinct-files
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// REDEFINE: %{merge_inputs} = %t/distinct-files/a_apply/a.edits.yaml %t/distinct-files/b_apply/b.edits.yaml
// RUN: %{merge}

// Both survive, different files:
// RUN: FileCheck --check-prefix=DF_MERGED --input-file=%t/distinct-files/merge_apply/merged.yaml %s
// DF_MERGED: FilePath:
// DF_MERGED-SAME: a.h
// DF_MERGED: FilePath:
// DF_MERGED-SAME: b.h

// RUN: FileCheck --check-prefix=DF_NOSTDERR --input-file=%t/distinct-files/merge.stderr --allow-empty %s
// DF_NOSTDERR-NOT: conflict:

// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=DF_A_APPLIED --input-file=%t/distinct-files/a.h %s %}
// DF_A_APPLIED: bounded_ptr<int> ga;
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=DF_B_APPLIED --input-file=%t/distinct-files/b.h %s %}
// DF_B_APPLIED: bounded_ptr<int> gb;


//--- distinct-locations/shared.h
int *g1;
int *g2;

//--- distinct-locations/a.cpp
#include "shared.h"
void use_a() { g1[5] = 0; }

//--- distinct-locations/b.cpp
#include "shared.h"
void use_b() { g2[7] = 0; }

// REDEFINE: %{casedir} = %t/distinct-locations
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// REDEFINE: %{merge_inputs} = %t/distinct-locations/a_apply/a.edits.yaml %t/distinct-locations/b_apply/b.edits.yaml
// RUN: %{merge}

// Both survive, same file, distinct offsets:
// RUN: FileCheck --check-prefix=DL_MERGED --input-file=%t/distinct-locations/merge_apply/merged.yaml %s
// DL_MERGED: Offset:          0
// DL_MERGED-NEXT: Length:          5
// DL_MERGED-NEXT: ReplacementText: 'bounded_ptr<int> '
// DL_MERGED: Offset:          9
// DL_MERGED-NEXT: Length:          5
// DL_MERGED-NEXT: ReplacementText: 'bounded_ptr<int> '

// RUN: FileCheck --check-prefix=DL_NOSTDERR --input-file=%t/distinct-locations/merge.stderr --allow-empty %s
// DL_NOSTDERR-NOT: conflict:

// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=DL_APPLIED --input-file=%t/distinct-locations/shared.h %s %}
// DL_APPLIED: bounded_ptr<int> g1;
// DL_APPLIED-NEXT: bounded_ptr<int> g2;


//--- three-tu-distinct/shared.h
int *g1;int *g2;int *g3;

//--- three-tu-distinct/a.cpp
#include "shared.h"
void use_a() { g1[5] = 0; }

//--- three-tu-distinct/b.cpp
#include "shared.h"
void use_b() { g2[7] = 0; }

//--- three-tu-distinct/c.cpp
#include "shared.h"
void use_c() { g3[9] = 0; }

// REDEFINE: %{casedir} = %t/three-tu-distinct
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}
// REDEFINE: %{testname} = c
// RUN: %{pipeline}

// REDEFINE: %{merge_inputs} = %t/three-tu-distinct/a_apply/a.edits.yaml %t/three-tu-distinct/b_apply/b.edits.yaml %t/three-tu-distinct/c_apply/c.edits.yaml
// RUN: %{merge}

// All three survive:
// RUN: FileCheck --check-prefix=TTD_MERGED --input-file=%t/three-tu-distinct/merge_apply/merged.yaml %s
// TTD_MERGED: Offset:          0
// TTD_MERGED: Offset:          8
// TTD_MERGED: Offset:          16

// RUN: FileCheck --check-prefix=TTD_NOSTDERR --input-file=%t/three-tu-distinct/merge.stderr --allow-empty %s
// TTD_NOSTDERR-NOT: conflict:

// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=TTD_APPLIED --input-file=%t/three-tu-distinct/shared.h %s %}
// TTD_APPLIED: bounded_ptr<int> g1;bounded_ptr<int> g2;bounded_ptr<int> g3;


//--- static-inline-fun-dedup/shared.h
static inline int * foo(int *p) {
  return p;
}

//--- static-inline-fun-dedup/a.cpp
#include "shared.h"
void caller(int * p) { int * q = foo(p); q[5] = 0; }

//--- static-inline-fun-dedup/b.cpp
#include "shared.h"
void caller(int * p) { int * q = foo(p); q[5] = 0; }

// REDEFINE: %{casedir} = %t/static-inline-fun-dedup
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// REDEFINE: %{merge_inputs} = %t/static-inline-fun-dedup/a_apply/a.edits.yaml %t/static-inline-fun-dedup/b_apply/b.edits.yaml
// RUN: %{merge}

// Shared static-linkage function's rewrite dedups; each caller's own file
// still survives on its own:
// RUN: FileCheck --check-prefix=SIF_MERGED --input-file=%t/static-inline-fun-dedup/merge_apply/merged.yaml %s
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: a.cpp
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: a.cpp
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: b.cpp
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: b.cpp
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: shared.h
// SIF_MERGED-NEXT: Offset:          14
// SIF_MERGED-NEXT: Length:          5
// SIF_MERGED-NEXT: ReplacementText: 'bounded_ptr<int> '
// SIF_MERGED: FilePath:
// SIF_MERGED-SAME: shared.h
// SIF_MERGED-NEXT: Offset:          24
// SIF_MERGED-NEXT: Length:          5
// SIF_MERGED-NEXT: ReplacementText: 'bounded_ptr<int> '
// SIF_MERGED-NOT: FilePath:

// RUN: FileCheck --check-prefix=SIF_NOSTDERR --input-file=%t/static-inline-fun-dedup/merge.stderr --allow-empty %s
// SIF_NOSTDERR-NOT: conflict:

// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=SIF_H_APPLIED --input-file=%t/static-inline-fun-dedup/shared.h %s %}
// SIF_H_APPLIED: static inline bounded_ptr<int> foo(bounded_ptr<int> p) {
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=SIF_A_APPLIED --input-file=%t/static-inline-fun-dedup/a.cpp %s %}
// SIF_A_APPLIED: void caller(bounded_ptr<int> p) { bounded_ptr<int> q = foo(p); q[5] = 0; }
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=SIF_B_APPLIED --input-file=%t/static-inline-fun-dedup/b.cpp %s %}
// SIF_B_APPLIED: void caller(bounded_ptr<int> p) { bounded_ptr<int> q = foo(p); q[5] = 0; }



