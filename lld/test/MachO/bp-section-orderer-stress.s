# REQUIRES: aarch64

# Generate a large test case and check that the output is deterministic.

# RUN: %python %s %t.s %t.proftext

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t.s -o %t.o
# RUN: llvm-profdata merge %t.proftext -o %t.profdata

# RUN: %lld -arch arm64 -lSystem -e _main --icf=all -o - %t.o --irpgo-profile-sort=%t.profdata --compression-sort=both | llvm-nm --numeric-sort --format=just-symbols - > %t.order1.txt
# RUN: %lld -arch arm64 -lSystem -e _main --icf=all -o - %t.o --irpgo-profile-sort=%t.profdata --compression-sort=both | llvm-nm --numeric-sort --format=just-symbols - > %t.order2.txt
# RUN: diff %t.order1.txt %t.order2.txt

import random
import sys

assembly_filepath = sys.argv[1]
proftext_filepath = sys.argv[2]

random.seed(1234)
num_functions = 1000
num_data = 100
num_traces = 10

function_names = [f"f{n}" for n in range(num_functions)]
data_names = [f"d{n}" for n in range(num_data)]
profiled_functions = function_names[: int(num_functions / 2)]

function_contents = [
    f"""
{name}:
  .cfi_startproc
  .cfi_personality 155, _personality_{i % 5}
  .cfi_lsda 16, _exception{i % 3}
  add w0, w0, #{i % 4096}
  add w1, w1, #{i % 10}
  add w2, w0, #{i % 20}
  adrp x3, {name}@PAGE
  ret
  .cfi_endproc
"""
    for i, name in enumerate(function_names)
]

data_contents = [
      f"""
{name}:
  .ascii "s{i % 2}-{i % 3}-{i % 5}"
  .xword {name}
"""
    for i, name in enumerate(data_names)
]

trace_contents = [
    f"""
# Weight
1
{", ".join(random.sample(profiled_functions, len(profiled_functions)))}
"""
    for i in range(num_traces)
]

profile_contents = [
    f"""
{name}
# Func Hash:
{i}
# Num Counters:
1
# Counter Values:
1
"""
    for i, name in enumerate(profiled_functions)
]

with open(assembly_filepath, "w") as f:
    f.write(
        f"""
.text
.globl _main

_main:
  ret

_personality_0:
  ret
_personality_1:
  ret
_personality_2:
  ret
_personality_3:
  ret
_personality_4:
  ret

_exception0:
  .quad 0x4200

_exception1:
  .quad 0x4210

_exception2:
  .quad 0x4220

{"".join(function_contents)}

.data
{"".join(data_contents)}

.subsections_via_symbols
"""
    )

with open(proftext_filepath, "w") as f:
    f.write(
        f"""
:ir
:temporal_prof_traces

# Num Traces
{num_traces}
# Trace Stream Size:
{num_traces}

{"".join(trace_contents)}

{"".join(profile_contents)}
"""
    )
