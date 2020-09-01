# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# Test implicit trace file name
# RUN: ld.lld --time-trace --time-trace-granularity=0 -o %t1.elf %t.o
# RUN: cat %t1.elf.time-trace \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

# Test specified trace file name
# RUN: ld.lld --time-trace --time-trace-file=%t2.json --time-trace-granularity=0 -o %t2.elf %t.o
# RUN: cat %t2.json \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

# Test trace requested to stdout
# RUN: ld.lld --time-trace --time-trace-file=- --time-trace-granularity=0 -o %t3.elf %t.o \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

# CHECK:      "beginningOfTime": {{[0-9]{16},}}
# CHECK-NEXT: "traceEvents": [

# Check one event has correct fields
# CHECK:      "dur":
# CHECK-NEXT: "name":
# CHECK-NEXT: "ph":
# CHECK-NEXT: "pid":
# CHECK-NEXT: "tid":
# CHECK-NEXT: "ts":

# Check there is an ExecuteLinker event
# CHECK: "name": "ExecuteLinker"

# Check process_name entry field
# CHECK: "name": "ld.lld{{(.exe)?}}"
# CHECK: "name": "process_name"
# CHECK: "name": "thread_name"

.globl _start
_start:
  ret
