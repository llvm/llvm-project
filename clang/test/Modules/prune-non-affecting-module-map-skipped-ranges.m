// Check that the skipped ranges of a preprocessing record are serialized with
// the adjustment applied to every other source location, so that the presence
// of non-affecting module map files does not affect the contents of PCM files.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- a/module.modulemap
module a {}

//--- b/module.modulemap
module b {}

//--- c/module.modulemap
module c { header "c.h" }
//--- c/c.h
@import b;
#if 0
int skipped;
#endif

//--- tu.m
@import c;

//--- common-args.rsp
-fmodule-map-file=b/module.modulemap -fmodule-map-file=c/module.modulemap -fmodules -fmodules-cache-path=cache -fdisable-module-hash -detailed-preprocessing-record -fsyntax-only tu.m
//--- common-args.rsp-end

// The module map for "a" is not affecting and gets pruned, which shifts the
// source location offsets of everything written after it. Every other location
// is adjusted for that shift before serialization, but the skipped range is
// written raw, so it keeps the pre-pruning offset and c.pcm ends up depending
// on whether "a" was on the command line.
//
// RUN: %clang_cc1 -working-directory %t @%t/common-args.rsp
// RUN: mv %t/cache %t/cache-no-a
// RUN: %clang_cc1 -working-directory %t -fmodule-map-file=a/module.modulemap @%t/common-args.rsp
// RUN: mv %t/cache %t/cache-a
//
// RUN: diff %t/cache-no-a/c.pcm %t/cache-a/c.pcm
