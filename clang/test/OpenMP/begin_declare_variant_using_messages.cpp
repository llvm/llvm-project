// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify %s

namespace BEFORE_AND_1 {
void before_and_1();
}
namespace AFTER_AND_2 {
void after_and_2(); // expected-note {{'AFTER_AND_2::after_and_2' declared here}} expected-note {{'AFTER_AND_2::after_and_2' declared here}}
}
namespace ONLY_1 {
void only_1(); // expected-note {{'ONLY_1::only_1' declared here}}
}
namespace BEFORE_1_AND_2 {
void before_1_and_2();
}

using BEFORE_1_AND_2::before_1_and_2;
using BEFORE_AND_1::before_and_1;

void test_before() {
  before_and_1();
  after_and_2(); // expected-error {{use of undeclared identifier 'after_and_2'; did you mean 'AFTER_AND_2::after_and_2'?}}
  only_1(); // expected-error {{use of undeclared identifier 'only_1'; did you mean 'ONLY_1::only_1'?}}
  before_1_and_2();
}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
using BEFORE_1_AND_2::before_1_and_2;
using BEFORE_AND_1::before_and_1;
using ONLY_1::only_1;
void test_1() {
  before_and_1();
  after_and_2(); // expected-error {{use of undeclared identifier 'after_and_2'; did you mean 'AFTER_AND_2::after_and_2'?}}
  only_1();
  before_1_and_2();
}
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
using AFTER_AND_2::after_and_2;
using BEFORE_1_AND_2::before_1_and_2;
void test_2() {
  before_and_1();
  after_and_2();
  only_1();
  before_1_and_2();
}
#pragma omp end declare variant

void test_after() {
  before_and_1();
  after_and_2();
  only_1();
  before_1_and_2();
}

using AFTER_AND_2::after_and_2;

// Make sure:
//  - we do not see the ast nodes for the gpu kind
//  - we do not choke on the text in the kind(fpga) guarded scopes
//  - we pick the right cbefore_1_and_2ees
