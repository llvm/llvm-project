// REQUIRES: zstd

// Regression test for a magic-byte collision in the compressed offload bundle
// (CCOB) reader. To find the boundary between concatenated bundles the reader
// used to scan for the literal 4-byte magic "CCOB"; those bytes can appear by
// chance inside a compressed payload, which truncated a valid single bundle and
// made decompression fail with "Src size is incorrect".
//
// Inputs/clang-offload-bundler-magic-collision.co is a single compressed bundle
// whose compressed region embeds the bytes "CCOB" inside a zstd skippable frame
// (regenerate with Inputs/clang-offload-bundler-magic-collision.py). The frame
// is ignored by the decompressor, so the bundle is valid, but a naive
// find("CCOB", 4) scan stops at the planted bytes. A reader that advances by
// the header's FileSize handles this file correctly.

// RUN: clang-offload-bundler -type=bc -list \
// RUN:   -input=%S/Inputs/clang-offload-bundler-magic-collision.co \
// RUN:   | FileCheck %s --check-prefix=LIST
// LIST-DAG: hip-amdgcn-amd-amdhsa--gfx906
// LIST-DAG: hip-amdgcn-amd-amdhsa--gfx908

// RUN: clang-offload-bundler -type=bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -input=%S/Inputs/clang-offload-bundler-magic-collision.co \
// RUN:   -output=%t.gfx906
// RUN: FileCheck %s --check-prefix=EXTRACT --input-file=%t.gfx906
// EXTRACT: Content of device file 1
