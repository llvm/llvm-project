#!/usr/bin/env python

"""Generate skeletal functions with a variety .cfi_ directives.
The purpose is to produce object-file test inputs to lld with a
variety of compact unwind encodings.
"""
import random
import argparse
import string
from math import factorial
from itertools import permutations

lsda_n = 0
lsda_odds = 0.0
func_size_low = 0x10
func_size_high = 0x100
saved_regs = ["%r15", "%r14", "%r13", "%r12", "%rbx"]
saved_regs_combined = list(list(permutations(saved_regs, i))
                           for i in range(0,6))

def print_function(name: str):
  global lsda_odds
  have_lsda = (random.random() < lsda_odds)
  frame_size = random.randint(4, 64) * 16
  frame_offset = -random.randint(0, (frame_size/16 - 4)) * 16
  reg_count = random.randint(0, 4)
  reg_combo = random.randint(0, factorial(reg_count) - 1)
  regs_saved = saved_regs_combined[reg_count][reg_combo]
  global func_size_low, func_size_high
  func_size = random.randint(func_size_low, func_size_high) * 0x10
  func_size_high += 1
  if func_size_high % 0x10 == 0:
    func_size_low += 1

  print(f"""\
### {name} regs={reg_count} frame={frame_size} lsda={have_lsda} size={func_size}
    .section __TEXT,__text,regular,pure_instructions
    .p2align 4, 0x90
    .globl {name}
{name}:
    .cfi_startproc""")
  if have_lsda:
    global lsda_n
    lsda_n += 1
    print(f"""\
    .cfi_personality 155, ___gxx_personality_v0
    .cfi_lsda 16, Lexception{lsda_n}""")
  print(f"""\
    pushq %rbp
    .cfi_def_cfa_offset {frame_size}
    .cfi_offset %rbp, {frame_offset+(6*8)}
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp""")
  for i in range(reg_count):
    print(f".cfi_offset {regs_saved[i]}, {frame_offset+(i*8)}")
  print(f"""\
    .fill {func_size - 6}
    popq %rbp
    retq
    .cfi_endproc
""")

  if have_lsda:
    print(f"""\
    .section __TEXT,__gcc_except_tab
    .p2align 2
Lexception{lsda_n}:
    .space 0x10
""")
  return func_size

def random_seed():
  """Generate a seed that can easily be passsed back in via --seed=STRING"""
  return ''.join(random.choice(string.ascii_lowercase) for i in range(10))

def main():
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="""\
Function sizes begin small then monotonically increase.  The goal is
to produce early pages that are full and later pages that are less
than full, in order to test handling for both cases.  Full pages
contain the maximum of 1021 compact unwind entries for a total page
size = 4 KiB.

Use --pages=N or --functions=N to control the size of the output.
Default is --pages=2, meaning produce at least two full pages of
compact unwind entries, plus some more. The calculatation is sloppy.
""")
  parser.add_argument('--seed', type=str, default=random_seed(),
                      help='Seed the random number generator')
  parser.add_argument('--pages', type=int, default=2,
                      help='Number of compact-unwind pages')
  parser.add_argument('--functions', type=int, default=None,
                      help='Number of functions to generate')
  parser.add_argument('--encodings', type=int, default=127,
                      help='Maximum number of unique unwind encodings (default = 127)')
  parser.add_argument('--lsda', type=int, default=0,
                      help='Percentage of functions with personality & LSDA (default = 10')
  args = parser.parse_args()
  random.seed(args.seed)
  p2align = 14
  global lsda_odds
  lsda_odds = args.lsda / 100.0

  print(f"""\
### seed={args.seed} lsda={lsda_odds} p2align={p2align}
    .section __TEXT,__text,regular,pure_instructions
    .p2align {p2align}, 0x90
""")

  size = 0
  base = (1 << p2align)
  if args.functions:
    for n in range(args.functions):
      size += print_function(f"x{size+base:08x}")
  else:
    while size < (args.pages << 24):
      size += print_function(f"x{size+base:08x}")

  print(f"""\
    .section __TEXT,__text,regular,pure_instructions
    .globl _main
    .p2align 4, 0x90
_main:
    retq

    .p2align 4, 0x90
___gxx_personality_v0:
    retq
""")


if __name__ == '__main__':
  main()
