# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## Test implicit trace file name
# RUN: %lld --time-trace --time-trace-granularity=0 -o %t1.macho %t.o
# RUN: cat %t1.macho.time-trace \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

## Test specified trace file name, also test that `--time-trace` is not needed if the other two are used.
# RUN: %lld --time-trace-file=%t2.json --time-trace-granularity=0 -o %t2.macho %t.o
# RUN: cat %t2.json \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

## Test trace requested to stdout, also test that `--time-trace` is not needed if the other two are used.
# RUN: %lld --time-trace-file=- --time-trace-granularity=0 -o %t3.macho %t.o \
# RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
# RUN:   | FileCheck %s

# CHECK:      "beginningOfTime": {{[0-9]{16},}}
# CHECK-NEXT: "traceEvents": [

## Check one event has correct fields
# CHECK:      "dur":
# CHECK-NEXT: "name":
# CHECK-NEXT: "ph":
# CHECK-NEXT: "pid":
# CHECK-NEXT: "tid":
# CHECK-NEXT: "ts":

## Check there is an ExecuteLinker event
# CHECK: "name": "ExecuteLinker"

## Check process_name entry field
# CHECK: "name": "ld64.lld{{(.exe)?}}"
# CHECK: "name": "process_name"
# CHECK: "name": "thread_name"

.globl _main
_main:
  ret
