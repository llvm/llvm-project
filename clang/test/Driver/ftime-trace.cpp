// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace -ftime-trace-granularity=0 -o out %s
// RUN: cat out.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=new-name.json -ftime-trace-granularity=0 -o out %s
// RUN: cat new-name.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: mkdir dir1 dir2
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=dir1 -ftime-trace-granularity=0 -o out %s
// RUN: cat dir1/out.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=dir2/ -ftime-trace-granularity=0 -o out %s
// RUN: cat dir2/out.json \
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
// CHECK:      "name": "{{clang|llvm}}{{.*}}"
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
