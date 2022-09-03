// UNSUPPORTED: ps4, ps5, hexagon

// RUN: rm -rf %T/exe && mkdir %T/exe
// RUN: %clangxx -ftime-trace -ftime-trace-granularity=0 -o %T/exe/check-time-trace %s
// RUN: cat %T/exe/check-time-trace*.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -ftime-trace -ftime-trace-granularity=0 -o %T/check-time-trace %s
// RUN: cat %T/check-time-trace.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -ftime-trace=%T/new-name.json -ftime-trace-granularity=0 -o %T/check-time-trace %s
// RUN: cat %T/new-name.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: rm -rf %T/output1 && mkdir %T/output1
// RUN: %clangxx -S -ftime-trace=%T/output1 -ftime-trace-granularity=0 -o %T/check-time-trace %s
// RUN: cat %T/output1/check-time-trace.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: rm -rf %T/output2 && mkdir %T/output2
// RUN: %clangxx -S -ftime-trace=%T/output2/ -ftime-trace-granularity=0 -o %T/check-time-trace %s
// RUN: cat %T/output2/check-time-trace.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s

// CHECK:      "beginningOfTime": {{[0-9]{16},}}
// CHECK-NEXT: "traceEvents": [
// CHECK:      "args":
// CHECK:      "detail":
// CHECK:      "dur":
// CHECK:      "name":
// CHECK-NEXT: "ph":
// CHECK-NEXT: "pid":
// CHECK-NEXT: "tid":
// CHECK-NEXT: "ts":
// CHECK:      "name": "clang{{.*}}"
// CHECK:      "name": "process_name"
// CHECK:      "name": "thread_name"

template <typename T>
struct Struct {
  T Num;
};

int main() {
  Struct<int> S;

  return 0;
}
