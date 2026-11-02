// Check that emitting the __LLVM,__remarks section for an optimization record
// makes the record's path absolute.
//
// Reaching the absolutize call needs a Mach-O target, since no other object
// file format has a remarks section, plus a bitstream record, since that is
// the only format requesting the section by default.
//
// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t && mkdir -p %t

// RUN: cd %t && %clang_cc1 -triple arm64-apple-macosx -emit-obj %s \
// RUN:     -o bitstream.o -opt-record-file rel.bitstream -opt-record-format bitstream
// RUN: llvm-readobj --string-dump=__remarks %t/bitstream.o \
// RUN:   | FileCheck %s --check-prefix=BITSTREAM-PATH -DDIR=%t
// BITSTREAM-PATH: String dump of section '__remarks':
// BITSTREAM-PATH: ] [[DIR]]{{[\\/]}}rel.bitstream

// A YAML record only asks for the section when told to, and then absolutizes
// the path a second time while serializing the section's metadata.
// RUN: cd %t && %clang_cc1 -triple arm64-apple-macosx -emit-obj %s \
// RUN:     -o yaml.o -opt-record-file rel.yaml -opt-record-format yaml -mllvm -remarks-section
// RUN: llvm-readobj --string-dump=__remarks %t/yaml.o \
// RUN:   | FileCheck %s --check-prefix=YAML-PATH -DDIR=%t
// YAML-PATH:      String dump of section '__remarks':
// YAML-PATH:      ] [[DIR]]{{[\\/]}}rel.yaml
