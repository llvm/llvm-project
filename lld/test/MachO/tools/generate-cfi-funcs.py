#!/usr/bin/env python

"""Generate skeletal functions with a variety .cfi_ directives.
The purpose is to produce object-file test inputs to lld with a
variety of compact unwind encodings.
"""
from __future__ import print_function
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

def print_function(name):
  global lsda_odds
  have_lsda = (random.random() < lsda_odds)
  frame_size = random.randint(4, 64) * 16
  frame_offset = -random.randint(0, (frame_size/16 - 4)) * 16
  global func_size_low, func_size_high
  func_size = random.randint(func_size_low, func_size_high) * 0x10
  func_size_high += 1
  if func_size_high % 0x10 == 0:
    func_size_low += 1

  print("""\
### %s frame=%d lsda=%s size=%d
    .section __TEXT,__text,regular,pure_instructions
    .p2align 4, 0x90
    .globl %s
%s:
    .cfi_startproc""" % (
        name, frame_size, have_lsda, func_size, name, name))
  if have_lsda:
    global lsda_n
    lsda_n += 1
    print("""\
    .cfi_personality 155, ___gxx_personality_v0
    .cfi_lsda 16, Lexception%d""" % lsda_n)
  print("""\
    pushq %%rbp
    .cfi_def_cfa_offset %d
    .cfi_offset %%rbp, %d
    movq %%rsp, %%rbp
    .cfi_def_cfa_register %%rbp""" % (frame_size, frame_offset + 6*8))
  print("""\
    .fill %d
    popq %%rbp
    retq
    .cfi_endproc
""" % (func_size - 6))

  if have_lsda:
    print("""\
    .section __TEXT,__gcc_except_tab
    .p2align 2
Lexception%d:
    .space 0x10
""" % lsda_n)
  return func_size

def random_seed():
  """Generate a seed that can easily be passed back in via --seed=STRING"""
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
compact unwind entries, plus some more. The calculation is sloppy.
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

  print("""\
### seed=%s lsda=%f p2align=%d
    .section __TEXT,__text,regular,pure_instructions
    .p2align %d, 0x90
""" % (args.seed, lsda_odds, p2align, p2align))

  size = 0
  base = (1 << p2align)
  if args.functions:
    for n in range(args.functions):
      size += print_function("x%08x" % (size+base))
  else:
    while size < (args.pages << 24):
      size += print_function("x%08x" % (size+base))

  print("""\
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
