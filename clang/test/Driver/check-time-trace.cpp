// RUN: %clangxx -S -ftime-trace -ftime-trace-granularity=0 -o %T/check-time-trace %s
// RUN: cat %T/check-time-trace.json \
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
