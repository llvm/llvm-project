#!/usr/bin/env python3
#===- lib/fuzzer/scripts/merge_data_flow.py ------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  return res.tostring().decode('utf-8')

def main(argv):
  D = {}
  C = {}
  # read the lines.
  for line in fileinput.input():
    # collect the coverage.
    if line.startswith('C'):
      COV = line.strip().split(' ')
      F = COV[0];
      if not F in C:
        C[F] = {0}
      for B in COV[1:]:
        C[F].add(int(B))
      continue
    # collect the data flow trace.
    [F,BV] = line.strip().split(' ')
    if F in D:
      D[F] = Merge(D[F], BV)
    else:
      D[F] = BV;
  # print the combined data flow trace.
  for F in D.keys():
    if isinstance(D[F], str):
      value = D[F]
    else:
      value = D[F].decode('utf-8')
    print("%s %s" % (F, value))
  # print the combined coverage
  for F in C.keys():
    print("%s" % F, end="")
    for B in list(C[F])[1:]:
      print(" %s" % B, end="")
    print()

if __name__ == '__main__':
  main(sys.argv)
