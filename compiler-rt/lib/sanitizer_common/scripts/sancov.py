#!/usr/bin/env python
# Merge or print the coverage data collected by asan's coverage.
# Input files are sequences of 4-byte integers.
# We need to merge these integers into a set and then
# either print them (as hex) or dump them into another file.
import array
import bisect
import glob
import os.path
import struct
import subprocess
import sys

prog_name = ""

def Usage():
  print >> sys.stderr, "Usage: \n" + \
      " " + prog_name + " merge FILE [FILE...] > OUTPUT\n" \
      " " + prog_name + " print FILE [FILE...]\n" \
      " " + prog_name + " unpack FILE [FILE...]\n" \
      " " + prog_name + " rawunpack FILE [FILE ...]\n" \
      " " + prog_name + " missing BINARY < LIST_OF_PCS\n"
  exit(1)

def CheckBits(bits):
  if bits != 32 and bits != 64:
    raise Exception("Wrong bitness: %d" % bits)

def TypeCodeForBits(bits):
  CheckBits(bits)
  return 'L' if bits == 64 else 'I'

def TypeCodeForStruct(bits):
  CheckBits(bits)
  return 'Q' if bits == 64 else 'I'

kMagic32SecondHalf = 0xFFFFFF32;
kMagic64SecondHalf = 0xFFFFFF64;
kMagicFirstHalf    = 0xC0BFFFFF;

def MagicForBits(bits):
  CheckBits(bits)
  if sys.byteorder == 'little':
    return [kMagic64SecondHalf if bits == 64 else kMagic32SecondHalf, kMagicFirstHalf]
  else:
    return [kMagicFirstHalf, kMagic64SecondHalf if bits == 64 else kMagic32SecondHalf]

def ReadMagicAndReturnBitness(f, path):
  magic_bytes = f.read(8)
  magic_words = struct.unpack('II', magic_bytes);
  bits = 0
  idx = 1 if sys.byteorder == 'little' else 0
  if magic_words[idx] == kMagicFirstHalf:
    if magic_words[1-idx] == kMagic64SecondHalf:
      bits = 64
    elif magic_words[1-idx] == kMagic32SecondHalf:
      bits = 32
  if bits == 0:
    raise Exception('Bad magic word in %s' % path)
  return bits

def ReadOneFile(path):
  with open(path, mode="rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    if size < 8:
      raise Exception('File %s is short (< 8 bytes)' % path)
    bits = ReadMagicAndReturnBitness(f, path)
    size -= 8
    s = struct.unpack_from(TypeCodeForStruct(bits) * (size * 8 / bits), f.read(size))
  print >>sys.stderr, "%s: read %d %d-bit PCs from %s" % (prog_name, size * 8 / bits, bits, path)
  return s

def Merge(files):
  s = set()
  for f in files:
    s = s.union(set(ReadOneFile(f)))
  print >> sys.stderr, "%s: %d files merged; %d PCs total" % \
    (prog_name, len(files), len(s))
  return sorted(s)

def PrintFiles(files):
  if len(files) > 1:
    s = Merge(files)
  else:  # If there is just on file, print the PCs in order.
    s = ReadOneFile(files[0])
    print >> sys.stderr, "%s: 1 file merged; %d PCs total" % \
      (prog_name, len(s))
  for i in s:
    print "0x%x" % i

def MergeAndPrint(files):
  if sys.stdout.isatty():
    Usage()
  s = Merge(files)
  bits = 32
  if max(s) > 0xFFFFFFFF:
    bits = 64
  array.array('I', MagicForBits(bits)).tofile(sys.stdout)
  a = struct.pack(TypeCodeForStruct(bits) * len(s), *s)
  sys.stdout.write(a)


def UnpackOneFile(path):
  with open(path, mode="rb") as f:
    print >> sys.stderr, "%s: unpacking %s" % (prog_name, path)
    while True:
      header = f.read(12)
      if not header: return
      if len(header) < 12:
        break
      pid, module_length, blob_size = struct.unpack('iII', header)
      module = f.read(module_length)
      blob = f.read(blob_size)
      assert(len(module) == module_length)
      assert(len(blob) == blob_size)
      extracted_file = "%s.%d.sancov" % (module, pid)
      print >> sys.stderr, "%s: extracting %s" % \
        (prog_name, extracted_file)
      # The packed file may contain multiple blobs for the same pid/module
      # pair. Append to the end of the file instead of overwriting.
      with open(extracted_file, 'ab') as f2:
        f2.write(blob)
    # fail
    raise Exception('Error reading file %s' % path)


def Unpack(files):
  for f in files:
    UnpackOneFile(f)

def UnpackOneRawFile(path, map_path):
  mem_map = []
  with open(map_path, mode="rt") as f_map:
    print >> sys.stderr, "%s: reading map %s" % (prog_name, map_path)
    bits = int(f_map.readline())
    if bits != 32 and bits != 64:
      raise Exception('Wrong bits size in the map')
    for line in f_map:
      parts = line.rstrip().split()
      mem_map.append((int(parts[0], 16),
                  int(parts[1], 16),
                  int(parts[2], 16),
                  ' '.join(parts[3:])))
  mem_map.sort(key=lambda m : m[0])
  mem_map_keys = [m[0] for m in mem_map]

  with open(path, mode="rb") as f:
    print >> sys.stderr, "%s: unpacking %s" % (prog_name, path)

    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    pcs = struct.unpack_from(TypeCodeForStruct(bits) * (size * 8 / bits), f.read(size))
    mem_map_pcs = [[] for i in range(0, len(mem_map))]

    for pc in pcs:
      if pc == 0: continue
      map_idx = bisect.bisect(mem_map_keys, pc) - 1
      (start, end, base, module_path) = mem_map[map_idx]
      assert pc >= start
      if pc >= end:
        print >> sys.stderr, "warning: %s: pc %x outside of any known mapping" % (prog_name, pc)
        continue
      mem_map_pcs[map_idx].append(pc - base)

    for ((start, end, base, module_path), pc_list) in zip(mem_map, mem_map_pcs):
      if len(pc_list) == 0: continue
      assert path.endswith('.sancov.raw')
      dst_path = module_path + '.' + os.path.basename(path)[:-4]
      print >> sys.stderr, "%s: writing %d PCs to %s" % (prog_name, len(pc_list), dst_path)
      sorted_pc_list = sorted(pc_list)
      pc_buffer = struct.pack(TypeCodeForStruct(bits) * len(pc_list), *sorted_pc_list)
      with open(dst_path, 'ab+') as f2:
        array.array('I', MagicForBits(bits)).tofile(f2)
        f2.seek(0, 2)
        f2.write(pc_buffer)

def RawUnpack(files):
  for f in files:
    if not f.endswith('.sancov.raw'):
      raise Exception('Unexpected raw file name %s' % f)
    f_map = f[:-3] + 'map'
    UnpackOneRawFile(f, f_map)

def GetInstrumentedPCs(binary):
  # This looks scary, but all it does is extract all offsets where we call:
  # - __sanitizer_cov() or __sanitizer_cov_with_check(),
  # - with call or callq,
  # - directly or via PLT.
  cmd = "objdump -d %s | " \
        "grep '^\s\+[0-9a-f]\+:.*\scall\(q\|\)\s\+[0-9a-f]\+ <__sanitizer_cov\(_with_check\|\)\(@plt\|\)>' | " \
        "grep '^\s\+[0-9a-f]\+' -o" % binary
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          shell=True)
  proc.stdin.close()
  # The PCs we get from objdump are off by 4 bytes, as they point to the
  # beginning of the callq instruction. Empirically this is true on x86 and
  # x86_64.
  return set(int(line.strip(), 16) + 4 for line in proc.stdout)

def PrintMissing(binary):
  if not os.path.isfile(binary):
    raise Exception('File not found: %s' % binary)
  instrumented = GetInstrumentedPCs(binary)
  print >> sys.stderr, "%s: found %d instrumented PCs in %s" % (prog_name,
                                                                len(instrumented),
                                                                binary)
  covered = set(int(line, 16) for line in sys.stdin)
  print >> sys.stderr, "%s: read %d PCs from stdin" % (prog_name, len(covered))
  missing = instrumented - covered
  print >> sys.stderr, "%s: %d PCs missing from coverage" % (prog_name, len(missing))
  if (len(missing) > len(instrumented) - len(covered)):
    print >> sys.stderr, \
        "%s: WARNING: stdin contains PCs not found in binary" % prog_name
  for pc in sorted(missing):
    print "0x%x" % pc

if __name__ == '__main__':
  prog_name = sys.argv[0]
  if len(sys.argv) <= 2:
    Usage();

  if sys.argv[1] == "missing":
    if len(sys.argv) != 3:
      Usage()
    PrintMissing(sys.argv[2])
    exit(0)

  file_list = []
  for f in sys.argv[2:]:
    file_list += glob.glob(f)
  if not file_list:
    Usage()

  if sys.argv[1] == "print":
    PrintFiles(file_list)
  elif sys.argv[1] == "merge":
    MergeAndPrint(file_list)
  elif sys.argv[1] == "unpack":
    Unpack(file_list)
  elif sys.argv[1] == "rawunpack":
    RawUnpack(file_list)
  else:
    Usage()
