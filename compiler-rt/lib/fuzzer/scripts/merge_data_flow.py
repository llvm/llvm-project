#!/usr/bin/env python
#===- lib/fuzzer/scripts/merge_data_flow.py ------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
# Merge several data flow traces into one.
# Usage:
#   merge_data_flow.py trace1 trace2 ...  > result
#===------------------------------------------------------------------------===#
import sys
import fileinput
from array import array

def Merge(a, b):
  res = array('b')
  for i in range(0, len(a)):
    res.append(ord('1' if a[i] == '1' or b[i] == '1' else '0'))
  return res.tostring()

def main(argv):
  D = {}
  for line in fileinput.input():
    [F,BV] = line.strip().split(' ')
    if F in D:
      D[F] = Merge(D[F], BV)
    else:
      D[F] = BV;
  for F in D.keys():
    print F, D[F]

if __name__ == '__main__':
  main(sys.argv)
