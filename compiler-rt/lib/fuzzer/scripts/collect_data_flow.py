#!/usr/bin/env python
#===- lib/fuzzer/scripts/collect_data_flow.py ------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
# Runs the data-flow tracer several times on the same input in order to collect
# the complete trace for all input bytes (running it on all bytes at once
# may fail if DFSan runs out of labels).
# Usage:
#
#   # Collect dataflow for one input, store it in OUTPUT (default is stdout)
#   collect_data_flow.py BINARY INPUT [OUTPUT]
#
#   # Collect dataflow for all inputs in CORPUS_DIR, store them in OUTPUT_DIR
#   collect_data_flow.py BINARY CORPUS_DIR OUTPUT_DIR
#===------------------------------------------------------------------------===#
import atexit
import hashlib
import sys
import os
import subprocess
import tempfile
import shutil

tmpdir = ""

def cleanup(d):
  print("removing: %s" % d)
  shutil.rmtree(d)

def collect_dataflow_for_corpus(self, exe, corpus_dir, output_dir):
  print("Collecting dataflow for corpus: %s output_dir: %s" % (corpus_dir,
                                                               output_dir))
  assert not os.path.exists(output_dir)
  os.mkdir(output_dir)
  for root, dirs, files in os.walk(corpus_dir):
    for f in files:
      path = os.path.join(root, f)
      sha1 = hashlib.sha1(open(path).read()).hexdigest()
      output = os.path.join(output_dir, sha1)
      subprocess.call([self, exe, path, output])
  functions_txt = open(os.path.join(output_dir, "functions.txt"), "w")
  subprocess.call([exe], stdout=functions_txt)


def main(argv):
  exe = argv[1]
  inp = argv[2]
  if os.path.isdir(inp):
    return collect_dataflow_for_corpus(argv[0], exe, inp, argv[3])
  size = os.path.getsize(inp)
  q = [[0, size]]
  tmpdir = tempfile.mkdtemp(prefix="libfuzzer-tmp-")
  atexit.register(cleanup, tmpdir)
  print "tmpdir: ", tmpdir
  outputs = []
  while len(q):
    r = q.pop()
    print "******* Trying:  ", r
    tmpfile = os.path.join(tmpdir, str(r[0]) + "-" + str(r[1]))
    ret = subprocess.call([exe, str(r[0]), str(r[1]), inp, tmpfile])
    if ret and r[1] - r[0] >= 2:
      q.append([r[0], (r[1] + r[0]) / 2])
      q.append([(r[1] + r[0]) / 2, r[1]])
    else:
      outputs.append(tmpfile)
      print "******* Success: ", r
  f = sys.stdout
  if len(argv) >= 4:
    f = open(argv[3], "w")
  merge = os.path.join(os.path.dirname(argv[0]), "merge_data_flow.py")
  subprocess.call([merge] + outputs, stdout=f)

if __name__ == '__main__':
  main(sys.argv)
