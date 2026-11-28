// TU mode alone must not run the analysis; it only changes how the analysis
// runs once some lifetime diagnostic (or the debug flag) enables it.
// Each analyzed function shows up as one "LifetimeSafetyAnalysis" time-trace
// event, so we count those.
//
// DEFINE: %{run} = %clang_cc1 -fsyntax-only \
// DEFINE:   -fexperimental-lifetime-safety-tu-analysis -Wno-return-stack-address \
// DEFINE:   -ftime-trace=%t.json -ftime-trace-granularity=0 %s
// DEFINE: %{count} = %python -c "import json, sys; \
// DEFINE:   print(sum(e['name'] == 'LifetimeSafetyAnalysis' \
// DEFINE:             for e in json.load(sys.stdin)['traceEvents']))" < %t.json
//
// RUN: %{run} -verify=none
// RUN: %{count} | FileCheck %s --check-prefix=NONE
// NONE: {{^}}0{{$}}
//
// RUN: %{run} -verify=none -fdebug-run-lifetime-safety
// RUN: %{count} | FileCheck %s --check-prefix=ALL
// ALL: {{^}}4{{$}}
//
// A lifetime diagnostic enabled by a pragma runs the analysis on the affected
// functions only.
// RUN: %{run} -DPRAGMA -verify=pragma
// RUN: %{count} | FileCheck %s --check-prefix=PRAGMA
// PRAGMA: {{^}}2{{$}}
//
// Inference runs the analysis on every function, so that callers analyzed later
// can rely on the annotations inferred for their callees.
// RUN: %{run} -flifetime-safety-inference -verify=none
// RUN: %{count} | FileCheck %s --check-prefix=ALL
// RUN: %{run} -flifetime-safety-inference -DPRAGMA -verify=pragma,infer
// RUN: %{count} | FileCheck %s --check-prefix=ALL

// none-no-diagnostics

int *id(int *p) { return p; }

#ifdef PRAGMA
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wlifetime-safety-return-stack-addr"
#endif
int *returnsLocal() {
  int local = 0;
  return &local; // pragma-warning {{stack memory associated with local variable 'local' is returned}} \
                 // pragma-note {{returned here}}
}

int *returnsLocalViaCallee() {
  int local = 0;
  return id(&local); // infer-warning {{stack memory associated with local variable 'local' is returned}} \
                     // infer-note {{returned here}}
}
#ifdef PRAGMA
#pragma clang diagnostic pop
#endif

int *returnsLocalOutside() {
  int local = 0;
  return &local; // no-warning
}
