// RUN: %clangxx -S -ftime-trace -ftime-trace-granularity=0 -o %T/check-time-trace-ParseDeclarationOrFunctionDefinition %s
// RUN: cat %T/check-time-trace-ParseDeclarationOrFunctionDefinition.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s

// CHECK-DAG: "name": "ParseDeclarationOrFunctionDefinition"
// CHECK-DAG: "detail": "{{.*}}check-time-trace-ParseDeclarationOrFunctionDefinition.cpp:15:1"
// CHECK-DAG: "name": "ParseFunctionDefinition"
// CHECK-DAG: "detail": "foo"
// CHECK-DAG: "name": "ParseFunctionDefinition"
// CHECK-DAG: "detail": "bar"

template <typename T>
void foo(T) {}
void bar() { foo(0); }
