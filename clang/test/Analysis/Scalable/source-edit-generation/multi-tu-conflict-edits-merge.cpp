// Tests clang-ssaf-src-edit-merge against real multi-TU source.
// Specifically, this file tests conflicting and duplicating source
// edits across multiple-TUs.

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
// DEFINE: %{merge} = clang-ssaf-src-edit-merge %{merge_inputs} -o %{casedir}/merged.yaml --sarif-conflicts-out=%{casedir}/conflicts.sarif 2> %{casedir}/merge.stderr

//--- dedup/shared.h
int *g;

//--- dedup/a.cpp
#include "shared.h"
void use_a() { g[5] = 0; }

//--- dedup/b.cpp
#include "shared.h"
void use_b() { g[7] = 0; }

// REDEFINE: %{casedir} = %t/dedup
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// Both TUs really produced the same replacement.
// RUN: FileCheck --check-prefix=DEDUP_A --input-file=%t/dedup/a_apply/a.edits.yaml %s
// DEDUP_A: Offset:          0
// DEDUP_A-NEXT: Length:          5
// DEDUP_A-NEXT: ReplacementText: 'bounded_ptr<int> '
// RUN: FileCheck --check-prefix=DEDUP_B --input-file=%t/dedup/b_apply/b.edits.yaml %s
// DEDUP_B: Offset:          0
// DEDUP_B-NEXT: Length:          5
// DEDUP_B-NEXT: ReplacementText: 'bounded_ptr<int> '

// REDEFINE: %{merge_inputs} = %t/dedup/a_apply/a.edits.yaml %t/dedup/b_apply/b.edits.yaml
// RUN: %{merge}

// Exactly one survivor:
// RUN: FileCheck --check-prefix=DEDUP_MERGED --input-file=%t/dedup/merged.yaml %s
// DEDUP_MERGED: Replacements:
// DEDUP_MERGED-NEXT: - FilePath:
// DEDUP_MERGED-SAME: shared.h
// DEDUP_MERGED-NEXT: Offset:          0
// DEDUP_MERGED-NEXT: Length:          5
// DEDUP_MERGED-NEXT: ReplacementText: 'bounded_ptr<int> '
// DEDUP_MERGED-NOT: FilePath:

// A dedup is not a conflict:
// RUN: FileCheck --check-prefix=DEDUP_NOSTDERR --input-file=%t/dedup/merge.stderr --allow-empty %s
// DEDUP_NOSTDERR-NOT: conflict:

// The SARIF document is still written (its presence just signals reporting
// was requested), but with no results.
// RUN: FileCheck --check-prefix=DEDUP_SARIF --input-file=%t/dedup/conflicts.sarif %s
// DEDUP_SARIF: "results": []


//--- two-way-conflict/shared.h
#if USE_LONG
long
#else
int
#endif
*g;

//--- two-way-conflict/a.cpp
#include "shared.h"
void use_a() { g[5] = 0; }

//--- two-way-conflict/b.cpp
#define USE_LONG 11
#include "shared.h"
void use_b() { g[7] = 0; }

// REDEFINE: %{casedir} = %t/two-way-conflict
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// Before merging: both real per-TU replacements exist and disagree.
// RUN: FileCheck --check-prefix=TWC_A --input-file=%t/two-way-conflict/a_apply/a.edits.yaml %s
// TWC_A: Offset:          24
// TWC_A-NEXT: Length:          12
// TWC_A-NEXT: ReplacementText: 'bounded_ptr<int> '
// RUN: FileCheck --check-prefix=TWC_B --input-file=%t/two-way-conflict/b_apply/b.edits.yaml %s
// TWC_B: Offset:          13
// TWC_B-NEXT: Length:          23
// TWC_B-NEXT: ReplacementText: 'bounded_ptr<long> '

// REDEFINE: %{merge_inputs} = %t/two-way-conflict/a_apply/a.edits.yaml %t/two-way-conflict/b_apply/b.edits.yaml
// RUN: %{merge}

// After merging: gone, and reported.
// RUN: FileCheck --check-prefix=TWC_MERGED --input-file=%t/two-way-conflict/merged.yaml %s
// TWC_MERGED: Replacements: []

// RUN: FileCheck --check-prefix=TWC_STDERR --input-file=%t/two-way-conflict/merge.stderr %s
// TWC_STDERR: conflict:
// TWC_STDERR-SAME: skipped
// TWC_STDERR-SAME: 2
// TWC_STDERR-SAME: shared.h:13

// RUN: FileCheck --check-prefix=TWC_SARIF --input-file=%t/two-way-conflict/conflicts.sarif %s
// TWC_SARIF: "level": "error"
// TWC_SARIF: "uri": "file://{{.*}}shared.h"
// TWC_SARIF: "byteOffset": 13
// TWC_SARIF: "relatedLocations":
// TWC_SARIF: "id": 1
// TWC_SARIF: "text": "candidate edit: \"bounded_ptr<int> \""
// TWC_SARIF: "id": 2
// TWC_SARIF: "text": "candidate edit: \"bounded_ptr<long> \""
// TWC_SARIF: "ruleId": "clang-reforge-replacement-conflict"



//--- three-way-conflict/shared.h
#if VARIANT == 1
long
#elif VARIANT == 2
char
#else
int
#endif
*g;

//--- three-way-conflict/a.cpp
#include "shared.h"
void use_a() { g[5] = 0; }

//--- three-way-conflict/b.cpp
#define VARIANT 1
#include "shared.h"
void use_b() { g[7] = 0; }

//--- three-way-conflict/c.cpp
#define VARIANT 2
#include "shared.h"
void use_c() { g[9] = 0; }

// REDEFINE: %{casedir} = %t/three-way-conflict
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// REDEFINE: %{testname} = b
// RUN: %{pipeline}
// REDEFINE: %{testname} = c
// RUN: %{pipeline}

// All three real per-TU replacements exist and disagree.
// RUN: FileCheck --check-prefix=THW_A --input-file=%t/three-way-conflict/a_apply/a.edits.yaml %s
// THW_A: Offset:          52
// THW_A-NEXT: Length:          12
// THW_A-NEXT: ReplacementText: 'bounded_ptr<int> '
// RUN: FileCheck --check-prefix=THW_B --input-file=%t/three-way-conflict/b_apply/b.edits.yaml %s
// THW_B: Offset:          17
// THW_B-NEXT: Length:          47
// THW_B-NEXT: ReplacementText: 'bounded_ptr<long> '
// RUN: FileCheck --check-prefix=THW_C --input-file=%t/three-way-conflict/c_apply/c.edits.yaml %s
// THW_C: Offset:          41
// THW_C-NEXT: Length:          23
// THW_C-NEXT: ReplacementText: 'bounded_ptr<char> '

// REDEFINE: %{merge_inputs} = %t/three-way-conflict/a_apply/a.edits.yaml %t/three-way-conflict/b_apply/b.edits.yaml %t/three-way-conflict/c_apply/c.edits.yaml
// RUN: %{merge}

// All three dropped as one cluster, not "two conflict and one wins".
// RUN: FileCheck --check-prefix=THW_MERGED --input-file=%t/three-way-conflict/merged.yaml %s
// THW_MERGED: Replacements: []

// RUN: FileCheck --check-prefix=THW_STDERR --input-file=%t/three-way-conflict/merge.stderr %s
// THW_STDERR: conflict:
// THW_STDERR-SAME: skipped
// THW_STDERR-SAME: 3
// THW_STDERR-SAME: shared.h:17

// The SARIF report names all three dropped candidates, not just two.
// RUN: FileCheck --check-prefix=THW_SARIF --input-file=%t/three-way-conflict/conflicts.sarif %s
// THW_SARIF: "level": "error"
// THW_SARIF: "uri": "file://{{.*}}shared.h"
// THW_SARIF: "byteOffset": 17
// THW_SARIF: "relatedLocations":
// THW_SARIF: "id": 1
// THW_SARIF: "text": "candidate edit: \"bounded_ptr<int> \""
// THW_SARIF: "id": 2
// THW_SARIF: "text": "candidate edit: \"bounded_ptr<char> \""
// THW_SARIF: "id": 3
// THW_SARIF: "text": "candidate edit: \"bounded_ptr<long> \""
// THW_SARIF: "ruleId": "clang-reforge-replacement-conflict"


// In order to create conflicts that differ only in text, copy 'v1.h'
// and 'v2.h' to 'shared.h' to create different content in 'shared.h'
// for 'a.cpp' and 'b.cpp'.

// FIXME: This test reveals a bug. If the array element edit is
// dropped, the bracket removal edit must be too.

//--- array-length-conflict/v1.h
int arr[3];

//--- array-length-conflict/v2.h
int arr[5];

//--- array-length-conflict/a.cpp
#include "shared.h"
void use_a() { arr[5] = 0; }

//--- array-length-conflict/b.cpp
#include "shared.h"
void use_b() { arr[7] = 0; }

// REDEFINE: %{casedir} = %t/array-length-conflict
// RUN: cp %t/array-length-conflict/v1.h %t/array-length-conflict/shared.h
// REDEFINE: %{testname} = a
// RUN: %{pipeline}
// RUN: cp %t/array-length-conflict/v2.h %t/array-length-conflict/shared.h
// REDEFINE: %{testname} = b
// RUN: %{pipeline}

// RUN: FileCheck --check-prefix=ARRLEN_A --input-file=%t/array-length-conflict/a_apply/a.edits.yaml %s
// ARRLEN_A: Offset:          0
// ARRLEN_A-NEXT: Length:          3
// ARRLEN_A-NEXT: ReplacementText: 'bounded_array<int, 3>'
// ARRLEN_A: Offset:          7
// ARRLEN_A-NEXT: Length:          3
// ARRLEN_A-NEXT: ReplacementText: ''
// RUN: FileCheck --check-prefix=ARRLEN_B --input-file=%t/array-length-conflict/b_apply/b.edits.yaml %s
// ARRLEN_B: Offset:          0
// ARRLEN_B-NEXT: Length:          3
// ARRLEN_B-NEXT: ReplacementText: 'bounded_array<int, 5>'
// ARRLEN_B: Offset:          7
// ARRLEN_B-NEXT: Length:          3
// ARRLEN_B-NEXT: ReplacementText: ''

// REDEFINE: %{merge_inputs} = %t/array-length-conflict/a_apply/a.edits.yaml %t/array-length-conflict/b_apply/b.edits.yaml
// RUN: %{merge}

// RUN: FileCheck --check-prefix=ARRLEN_MERGED --input-file=%t/array-length-conflict/merged.yaml %s
// ARRLEN_MERGED: Replacements:
// ARRLEN_MERGED-NEXT: - FilePath:
// ARRLEN_MERGED-SAME: shared.h
// ARRLEN_MERGED-NEXT: Offset:          7
// ARRLEN_MERGED-NEXT: Length:          3
// ARRLEN_MERGED-NEXT: ReplacementText: ''
// ARRLEN_MERGED-NOT: FilePath:


// RUN: FileCheck --check-prefix=ARRLEN_STDERR --input-file=%t/array-length-conflict/merge.stderr %s
// ARRLEN_STDERR: conflict:
// ARRLEN_STDERR-SAME: skipped
// ARRLEN_STDERR-SAME: 2
// ARRLEN_STDERR-SAME: shared.h:0
// ARRLEN_STDERR-NOT: conflict:

// RUN: FileCheck --check-prefix=ARRLEN_SARIF --input-file=%t/array-length-conflict/conflicts.sarif %s
// ARRLEN_SARIF: "level": "error"
// ARRLEN_SARIF: "uri": "file://{{.*}}shared.h"
// ARRLEN_SARIF: "byteOffset": 0
// ARRLEN_SARIF: "relatedLocations":
// ARRLEN_SARIF: "id": 1
// ARRLEN_SARIF: "text": "candidate edit: \"bounded_array<int, 3>\""
// ARRLEN_SARIF: "id": 2
// ARRLEN_SARIF: "text": "candidate edit: \"bounded_array<int, 5>\""
// ARRLEN_SARIF: "ruleId": "clang-reforge-replacement-conflict"


