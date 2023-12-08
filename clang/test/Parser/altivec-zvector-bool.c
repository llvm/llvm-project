// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix-xcoff \
// RUN:            -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple=powerpc-ibm-aix-xcoff \
// RUN:            -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple=powerpc-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple=s390x-linux-gnu -target-cpu arch11 \
// RUN:            -fzvector -fsyntax-only %s
// RUN: %clang_cc1 -triple=s390x-ibm-zos -target-cpu arch11 \
// RUN:            -fzvector -fsyntax-only %s

__vector bool char bc;
__vector bool short bsh;
__vector bool short int bshi;
__vector bool int bi;
__vector _Bool char bc;
__vector _Bool short bsh;
__vector _Bool short int bshi;
__vector _Bool int bi;
