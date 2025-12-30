// Check that the presence of non-affecting module map files does not affect the
// contents of PCM files.

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

//--- tu.m
@import c;

//--- explicit-mms-common-args.rsp
-fmodule-map-file=b/module.modulemap -fmodule-map-file=c/module.modulemap -fmodules -fmodules-cache-path=cache -fdisable-module-hash -fsyntax-only tu.m
//--- implicit-search-args.rsp
-I a -I b -I c -fimplicit-module-maps -fmodules -fmodules-cache-path=cache -fdisable-module-hash -fsyntax-only tu.m
//--- implicit-search-args.rsp-end

// Test with explicit module map files.
//
// RUN: %clang_cc1 -working-directory %t @%t/explicit-mms-common-args.rsp
// RUN: mv %t/cache %t/cache-explicit-no-a-prune
// RUN: %clang_cc1 -working-directory %t @%t/explicit-mms-common-args.rsp -fno-modules-prune-non-affecting-module-map-files
// RUN: mv %t/cache %t/cache-explicit-no-a-keep
//
// RUN: %clang_cc1 -working-directory %t -fmodule-map-file=a/module.modulemap @%t/explicit-mms-common-args.rsp
// RUN: mv %t/cache %t/cache-explicit-a-prune
// RUN: %clang_cc1 -working-directory %t -fmodule-map-file=a/module.modulemap @%t/explicit-mms-common-args.rsp -fno-modules-prune-non-affecting-module-map-files
// RUN: mv %t/cache %t/cache-explicit-a-keep
//
// RUN: diff %t/cache-explicit-no-a-prune/c.pcm %t/cache-explicit-a-prune/c.pcm
// RUN: not diff %t/cache-explicit-no-a-keep/c.pcm %t/cache-explicit-a-keep/c.pcm

// Test with implicit module map search.
//
// RUN: %clang_cc1 -working-directory %t @%t/implicit-search-args.rsp
// RUN: mv %t/cache %t/cache-implicit-no-a-prune
// RUN: %clang_cc1 -working-directory %t @%t/implicit-search-args.rsp -fno-modules-prune-non-affecting-module-map-files
// RUN: mv %t/cache %t/cache-implicit-no-a-keep
//
// FIXME: Instead of removing "a/module.modulemap" from the file system, we
//        could drop the "-I a" search path argument in combination with the
//        "-fmodules-skip-header-search-paths" flag. Unfortunately, that flag
//        does not prevent serialization of the search path usage bit vector,
//        making the files differ anyways.
// RUN: rm %t/a/module.modulemap
//
// RUN: %clang_cc1 -working-directory %t @%t/implicit-search-args.rsp
// RUN: mv %t/cache %t/cache-implicit-a-prune
// RUN: %clang_cc1 -working-directory %t @%t/implicit-search-args.rsp -fno-modules-prune-non-affecting-module-map-files
// RUN: mv %t/cache %t/cache-implicit-a-keep
//
// RUN: diff %t/cache-implicit-no-a-prune/c.pcm %t/cache-implicit-a-prune/c.pcm
// RUN: not diff %t/cache-implicit-no-a-keep/c.pcm %t/cache-implicit-a-keep/c.pcm
