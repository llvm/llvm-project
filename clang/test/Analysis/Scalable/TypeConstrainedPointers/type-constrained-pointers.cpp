// End-to-end test of the Type Constrained Pointers analysis.

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=TypeConstrainedPointers \
// RUN:   --ssaf-tu-summary-file=%t/tu.summary.json \
// RUN:   --ssaf-compilation-unit-id="tu-1"

// RUN: clang-ssaf-linker %t/tu.summary.json -o %t/lu.json

// RUN: clang-ssaf-analyzer %t/lu.json -o %t/wpa.json \
// RUN:   -a TypeConstrainedPointersAnalysisResult

// RUN: FileCheck %s --input-file=%t/wpa.json

typedef __SIZE_TYPE__ size_t;

// Plain operator new: only the return entity (suffix "0") is extracted.
void *operator new(size_t size);
void *operator new(size_t size) { return (void*)1; }

// Placement operator new: return entity (suffix "0") + 2nd param (suffix "2").
void *operator new(size_t size, void *placement) noexcept;
void *operator new(size_t size, void *placement) noexcept { return (void*)1; }

// Plain operator delete: 1st param (suffix "1") is extracted.
void operator delete(void *ptr) noexcept;
void operator delete(void *ptr) noexcept {}

// Placement operator delete: 1st param (suffix "1") + 2nd param (suffix "2").
void operator delete(void *ptr, void *placement) noexcept;
void operator delete(void *ptr, void *placement) noexcept {}

// main: argv (suffix "2") is extracted; argc (suffix "1") is not a pointer.
int main(int argc, char **argv);
int main(int argc, char **argv) { return 0; }

// Plain new: return entity (suffix "0").
// CHECK-DAG: "id": [[NEW_RET_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "0",[[:space:]]+"usr": }}"c:@F@operator new#{{.*}}#"

// Placement new: return entity (suffix "0") and placement param (suffix "2").
// CHECK-DAG: "id": [[NEW_PLACE_RET_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "0",[[:space:]]+"usr": }}"c:@F@operator new#{{.*}}#*v#"
// CHECK-DAG: "id": [[NEW_PLACE_PARAM_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "2",[[:space:]]+"usr": }}"c:@F@operator new#{{.*}}#*v#"

// Plain delete: ptr param (suffix "1").
// CHECK-DAG: "id": [[DEL_PTR_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "1",[[:space:]]+"usr": }}"c:@F@operator delete#*v#"

// Placement delete: ptr param (suffix "1") and placement param (suffix "2").
// CHECK-DAG: "id": [[DEL_PLACE_PTR_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "1",[[:space:]]+"usr": }}"c:@F@operator delete#*v#S0_#"
// CHECK-DAG: "id": [[DEL_PLACE_PARAM_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "2",[[:space:]]+"usr": }}"c:@F@operator delete#*v#S0_#"

// main: argv (suffix "2", 0-based param index 1).
// CHECK-DAG: "id": [[MAIN_ARGV_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "2",[[:space:]]+"usr": }}"c:@F@main{{.*}}"

// CHECK: "analysis_name": "TypeConstrainedPointersAnalysisResult"

// CHECK-DAG: "@": [[NEW_RET_ID]]
// CHECK-DAG: "@": [[NEW_PLACE_RET_ID]]
// CHECK-DAG: "@": [[NEW_PLACE_PARAM_ID]]
// CHECK-DAG: "@": [[DEL_PTR_ID]]
// CHECK-DAG: "@": [[DEL_PLACE_PTR_ID]]
// CHECK-DAG: "@": [[DEL_PLACE_PARAM_ID]]
// CHECK-DAG: "@": [[MAIN_ARGV_ID]]

// CHECK: "type": "WPASuite"
