// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK1
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK1
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK1
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK1

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK2
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK2
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK2
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK2

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK3
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK3
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK3
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK3

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK4
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK4
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK4
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK4

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK5
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK5
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK5
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK5

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK6
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK6
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK6
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK6

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK7
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s -check-prefix=CHECK7
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK7
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  -check-prefix=CHECK7


// expected-no-diagnostics


#ifndef HEADER
#define HEADER

// COM: Checking for default target and thread limit: CHECK1

void default_val_num_teams() {
    #pragma omp target simd
    for (int i = 0; i < 22; i++)
        int a_var;
}

// COM: Check for num_teams(22) all the same: CHECK2
void foo1() {
    #pragma omp target teams num_teams(22)
    { int a_var; }
}

void foo2() {
    #pragma omp target teams distribute num_teams(22)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void foo3() {
    #pragma omp target teams distribute parallel for num_teams(22)
    for (int i = 0; i < 22; i++)
        int a_var;
}

// COM: Check for num_teams differently: CHECK3
void bar1() {
    #pragma omp target teams num_teams(22)
    { int a_var; }
}

void bar2() {
    #pragma omp target teams distribute num_teams(33)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void bar3() {
    #pragma omp target teams distribute parallel for num_teams(44)
    for (int i = 0; i < 22; i++)
        int a_var;
}

// COM: Check for const int expression: CHECK4
void const_int() {
    const int NT = 22;
    #pragma omp target teams num_teams(NT)
    { int a_var; }
}

// COM: Checking for num threads: CHECK5
void thread_limit() {
    #pragma omp target teams thread_limit(22)
    { int a_var; }
}

// COM: Checking for num threads and thread limit: CHECK6
void num_threads() {
    #pragma omp target teams distribute parallel for thread_limit(22) num_threads(11)
    for (int i = 0; i < 22; i++)
        int a_var;
}

// COM: Checking for thread_limit and num_teams: CHECK6
void threads_and_teams() {
    #pragma omp target teams distribute parallel for thread_limit(22) num_teams(33)
    for (int i = 0; i < 22; i++)
        int a_var;
}

#endif

// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}default_val_num_teams{{[^(]*}}
// CHECK1-SAME: () #[[ATTR_OUTLINED_DEF_SIMD:[0-9]+]] {

// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}foo1{{[^(]*}}
// CHECK2-SAME: () #[[ATTR_OUTLINED_CHECK2:[0-9]+]] {
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}foo2{{[^(]*}}
// CHECK2-SAME: () #[[ATTR_OUTLINED_CHECK2:[0-9]+]] {
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}foo3{{[^(]*}}
// CHECK2-SAME: () #[[ATTR_OUTLINED_CHECK2:[0-9]+]] {

// CHECK3-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}bar1{{[^(]*}}
// CHECK3-SAME: () #[[ATTR_OUTLINED_CHECK3_1:[0-9]+]] {
// CHECK3-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}bar2{{[^(]*}}
// CHECK3-SAME: () #[[ATTR_OUTLINED_CHECK3_2:[0-9]+]] {
// CHECK3-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}bar3{{[^(]*}}
// CHECK3-SAME: () #[[ATTR_OUTLINED_CHECK3_3:[0-9]+]] {

// CHECK4-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}const_int{{[^(]*}}
// CHECK4-SAME: () #[[ATTR_OUTLINED_CHECK4:[0-9]+]] {

// CHECK5-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}thread_limit{{[^(]*}}
// CHECK5-SAME: () #[[ATTR_OUTLINED_CHECK5:[0-9]+]] {

// CHECK6-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}num_threads{{[^(]*}}
// CHECK6-SAME: () #[[ATTR_OUTLINED_CHECK6:[0-9]+]] {

// CHECK7-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+.*}}threads_and_teams{{[^(]*}}
// CHECK7-SAME: () #[[ATTR_OUTLINED_CHECK7:[0-9]+]] {



// CHECK1: attributes #[[ATTR_OUTLINED_DEF_SIMD]] = {{{.*"omp_target_num_teams"="1".*"omp_target_thread_limit"="1".*}}}

// CHECK2: attributes #[[ATTR_OUTLINED_CHECK2]] = {{{.*"omp_target_num_teams"="22".*}}}

// CHECK3: attributes #[[ATTR_OUTLINED_CHECK3_1]] = {{{.*"omp_target_num_teams"="22".*}}}
// CHECK3: attributes #[[ATTR_OUTLINED_CHECK3_2]] = {{{.*"omp_target_num_teams"="33".*}}}
// CHECK3: attributes #[[ATTR_OUTLINED_CHECK3_3]] = {{{.*"omp_target_num_teams"="44".*}}}

// CHECK4: attributes #[[ATTR_OUTLINED_CHECK4]] = {{{.*"omp_target_num_teams"="22".*}}}

// CHECK5: attributes #[[ATTR_OUTLINED_CHECK5]] = {{{.*"omp_target_thread_limit"="22".*}}}

// CHECK6: attributes #[[ATTR_OUTLINED_CHECK6]] = {{{.*"omp_target_thread_limit"="11".*}}}

// CHECK7: attributes #[[ATTR_OUTLINED_CHECK7]] = {{{.*"omp_target_num_teams"="33".*"omp_target_thread_limit"="22".*}}}
