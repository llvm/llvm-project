// RUN: rm -rf %t

// A from path 1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-ignore-macro=EXPECTED_PATH -fmodules-ignore-macro=DIRECT -fsyntax-only %s -verify -I %S/Inputs/modules-with-same-name/path1/A -DDIRECT -DEXPECTED_PATH=1

// A from path 2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-ignore-macro=EXPECTED_PATH -fmodules-ignore-macro=DIRECT -fsyntax-only %s -verify -I %S/Inputs/modules-with-same-name/path2/A -DDIRECT -DEXPECTED_PATH=2

// Confirm that we have two pcm files (one for each 'A').
// RUN: find %t -name "A-*.pc[m]" | count 2

// DependsOnA, using A from path 1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-ignore-macro=EXPECTED_PATH -fmodules-ignore-macro=DIRECT -fsyntax-only %s -verify -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path1/A -DEXPECTED_PATH=1

// Confirm that we have three pcm files (one for each 'A', and one for 'DependsOnA')
// RUN: find %t -name "*.pc[m]" | count 3

// DependsOnA, using A from path 2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-ignore-macro=EXPECTED_PATH -fmodules-ignore-macro=DIRECT -fsyntax-only %s -verify -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path2/A -DEXPECTED_PATH=2

// Confirm that we still have three pcm files, since DependsOnA will be rebuilt
// RUN: find %t -name "*.pc[m]" | count 3

#ifdef DIRECT
@import A;
#else
@import DependsOnA;
#endif

#if FROM_PATH != EXPECTED_PATH
#error "Got the wrong module!"
#endif

// expected-no-diagnostics
