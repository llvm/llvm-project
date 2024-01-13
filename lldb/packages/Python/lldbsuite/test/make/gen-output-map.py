#!/bin/env python3
import json
import sys

args = sys.argv[1:]
if len(args) % 2:
    print("""
Usage: {0} file1.swift file2.swift ... file1.o file2.o ...
Generates output-file-map.json
    """.format(sys.argv[0]))
    exit(1)

midpoint = int(len(args) / 2)
sources = args[:midpoint]
objects = args[midpoint:]

# {
#   "": {
#     "swift-dependencies": "/tmp/overall.swiftdeps",
#   },
#   "main.swift": {
#     "object": "/tmp/main.o",
#     "swift-dependencies": "/tmp/main.swiftdeps",
#   },
# }

data = {}
for src, obj in zip(sources, objects):
    data[src] = {"object": obj}

json.dump(data, sys.stdout)
print("")
