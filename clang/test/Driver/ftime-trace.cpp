// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose -o out %s
// RUN: cat out.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=new-name.json -ftime-trace-granularity=0 -ftime-trace-verbose -o out %s
// RUN: cat new-name.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: mkdir dir1 dir2
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=dir1 -ftime-trace-granularity=0 -ftime-trace-verbose -o out %s
// RUN: cat dir1/out.json \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s
// RUN: %clangxx -S -no-canonical-prefixes -ftime-trace=dir2/ -ftime-trace-granularity=0 -ftime-trace-verbose -o out %s
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

// RUN: mkdir d e f && cp %s d/a.cpp && touch d/b.c

/// TODO: Support -fno-integrated-as.
// RUN: %clang -### -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose -fintegrated-as d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefix=COMPILE1
// COMPILE1: -cc1{{.*}} "-ftime-trace=e/a.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// RUN: %clang -### -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp d/b.c -dumpdir f/ 2>&1 | FileCheck %s --check-prefix=COMPILE2
// COMPILE2: -cc1{{.*}} "-ftime-trace=f/a.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// COMPILE2: -cc1{{.*}} "-ftime-trace=f/b.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

/// -o specifies the link output. Create ${output}-${basename}.json.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp d/b.c -o e/x 2>&1 | FileCheck %s --check-prefix=LINK1
// LINK1: -cc1{{.*}} "-ftime-trace=e/x-a.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK1: -cc1{{.*}} "-ftime-trace=e/x-b.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

/// -dumpdir is f/g, not ending with a path separator. We create f/g${basename}.json.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp d/b.c -o e/x -dumpdir f/g 2>&1 | FileCheck %s --check-prefix=LINK2
// LINK2: -cc1{{.*}} "-ftime-trace=f/ga.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK2: -cc1{{.*}} "-ftime-trace=f/gb.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// RUN: %clang -### -ftime-trace=e -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp d/b.c -o f/x -dumpdir f/ 2>&1 | FileCheck %s --check-prefix=LINK3
// LINK3: -cc1{{.*}} "-ftime-trace=e{{/|\\\\}}a-{{[^.]*}}.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK3: -cc1{{.*}} "-ftime-trace=e{{/|\\\\}}b-{{[^.]*}}.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// RUN: %clang -### -ftime-trace -ftime-trace=e -ftime-trace-granularity=1 -ftime-trace-verbose -xassembler d/a.cpp 2>&1 | \
// RUN:   FileCheck %s --check-prefix=UNUSED
// UNUSED:      warning: argument unused during compilation: '-ftime-trace'
// UNUSED-NEXT: warning: argument unused during compilation: '-ftime-trace=e'
// UNUSED-NEXT: warning: argument unused during compilation: '-ftime-trace-granularity=1'
// UNUSED-NEXT: warning: argument unused during compilation: '-ftime-trace-verbose'
// UNUSED-NOT:  warning:

template <typename T>
struct Struct {
  T Num;
};

int main() {
  Struct<int> S;

  return 0;
}
