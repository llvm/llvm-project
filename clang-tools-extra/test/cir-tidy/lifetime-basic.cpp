// RUN: %check_cir_tidy %s cir-lifetime-check %t \
// RUN: --export-fixes=%t.yaml \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: cir-lifetime-check.RemarksList, value: "all"}, \
// RUN:   {key: cir-lifetime-check.HistLimit, value: "1"}, \
// RUN:   {key: cir-lifetime-check.CodeGenBuildDeferredThreshold, value: "500"}, \
// RUN:   {key: cir-lifetime-check.CodeGenSkipFunctionsFromSystemHeaders, value: "false"}, \
// RUN:   {key: cir-lifetime-check.HistoryList, value: "invalid;null"}]}' \
// RUN: --
// RUN: FileCheck -input-file=%t.yaml -check-prefix=CHECK-YAML %s

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42; // CHECK-MESSAGES: remark: pset => { x }
  }        // CHECK-NOTES: note: pointee 'x' invalidated at end of scope
  *p = 42; // CHECK-MESSAGES: remark: pset => { invalid }
           // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: use of invalid pointer 'p'
  return p;
}

// CHECK-YAML:    DiagnosticMessage:
// CHECK-YAML:      Message:         'pset => { x }'
// CHECK-YAML:      Replacements:    []
// CHECK-YAML:    Level:           Remark

// CHECK-YAML:    DiagnosticMessage:
// CHECK-YAML:      Message:         'pset => { invalid }'
// CHECK-YAML:      Replacements:    []
// CHECK-YAML:    Level:           Remark

// CHECK-YAML: DiagnosticMessage:
// CHECK-YAML:   Message:         'use of invalid pointer ''p'''
// CHECK-YAML:   Replacements:    []
// CHECK-YAML: Notes:
// CHECK-YAML:   - Message:         'pointee ''x'' invalidated at end of scope'
// CHECK-YAML:     Replacements:    []
// CHECK-YAML: Level:           Warning