#!/usr/bin/env python

"""Validate compact unwind info by cross checking the llvm-objdump
reports of the input object file vs final linked output.
"""
from __future__ import print_function
import sys
import argparse
import re
from pprint import pprint

def main():
  hex = "[a-f\d]"
  hex8 = hex + "{8}"

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('files', metavar='FILES', nargs='*',
                      help='output of (llvm-objdump --unwind-info --syms) for object file(s) plus final linker output')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  if args.files:
    objdump_string = ''.join([open(f).read() for f in args.files])
  else:
    objdump_string = sys.stdin.read()

  object_encodings_list = [(symbol, encoding, personality, lsda)
    for symbol, encoding, personality, lsda in
    re.findall(r"start:\s+0x%s+\s+(\w+)\s+" % hex +
               r"length:\s+0x%s+\s+" % hex +
               r"compact encoding:\s+0x(%s+)(?:\s+" % hex +
               r"personality function:\s+0x(%s+)\s+\w+\s+" % hex +
               r"LSDA:\s+0x(%s+)\s+\w+(?: \+ 0x%s+)?)?" % (hex, hex),
               objdump_string, re.DOTALL)]
  object_encodings_map = {symbol:encoding
    for symbol, encoding, _, _ in object_encodings_list}
  if not object_encodings_map:
    sys.exit("no object encodings found in input")

  program_symbols_map = {address:symbol
    for address, symbol in
    re.findall(r"^%s(%s) g\s+F __TEXT,__text (x\1)$" % (hex8, hex8),
               objdump_string, re.MULTILINE)}
  if not program_symbols_map:
    sys.exit("no program symbols found in input")

  program_common_encodings = (
    re.findall(r"^\s+encoding\[(?:\d|\d\d|1[01]\d|12[0-6])\]: 0x(%s+)$" % hex,
               objdump_string, re.MULTILINE))
  if not program_common_encodings:
    sys.exit("no common encodings found in input")

  program_encodings_map = {program_symbols_map[address]:encoding
    for address, encoding in
    re.findall(r"^\s+\[\d+\]: function offset=0x(%s+), " % hex +
               r"encoding(?:\[\d+\])?=0x(%s+)$" % hex,
               objdump_string, re.MULTILINE)}
  if not object_encodings_map:
    sys.exit("no program encodings found in input")

  # Fold adjacent entries from the object file that have matching encodings
  # TODO(gkm) add check for personality+lsda
  encoding0 = 0
  for symbol in sorted(object_encodings_map):
    encoding = object_encodings_map[symbol]
    fold = (encoding == encoding0)
    if fold:
      del object_encodings_map[symbol]
    if args.debug:
      print("%s %s with %s" % (
              'delete' if fold else 'retain', symbol, encoding))
    encoding0 = encoding

  if program_encodings_map != object_encodings_map:
    if args.debug:
      print("program encodings map:")
      pprint(program_encodings_map)
      print("object encodings map:")
      pprint(object_encodings_map)
    sys.exit("encoding maps differ")

  # Count frequency of object-file folded encodings
  # and compare with the program-file common encodings table
  encoding_frequency_map = {}
  for _, encoding in object_encodings_map.items():
    encoding_frequency_map[encoding] = 1 + encoding_frequency_map.get(encoding, 0)
  encoding_frequencies = [x for x in
                          sorted(encoding_frequency_map,
                                 key=lambda x: (encoding_frequency_map.get(x), x),
                                 reverse=True)]
  del encoding_frequencies[127:]

  if program_common_encodings != encoding_frequencies:
    if args.debug:
      pprint("program common encodings:\n" + str(program_common_encodings))
      pprint("object encoding frequencies:\n" + str(encoding_frequencies))
    sys.exit("encoding frequencies differ")


if __name__ == '__main__':
  main()
