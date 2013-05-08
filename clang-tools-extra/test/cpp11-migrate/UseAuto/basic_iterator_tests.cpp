// This file contains basic positive tests for the use-auto transform's ability
// to replace standard iterators. Variables considered:
// * All std container names
// * All std iterator names
// * Different patterns of defining iterators and containers
//
// // The most basic test.
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=array -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
//
// Test variations on how the container and its iterators might be defined.
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=array \
// RUN:   -DUSE_INLINE_NAMESPACE -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=array \
// RUN:   -DUSE_BASE_CLASS_ITERATORS -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=array \
// RUN:   -DUSE_INNER_CLASS_ITERATORS -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
//
// Test all of the other container names in a basic configuration.
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=deque -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=forward_list -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=list -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=vector -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=map -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=multimap -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=set -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=multiset -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=unordered_map -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=unordered_multimap -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=unordered_set -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=unordered_multiset -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=queue -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=priority_queue -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -DCONTAINER=stack -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#ifndef CONTAINER
#error You must define CONTAINER to the name of the container for testing.
#endif

#include "test_std_container.h"

int main(int argc, char **argv) {
  {
    std::CONTAINER<int> C;
    std::CONTAINER<int>::iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }
  {
    std::CONTAINER<int> C;
    std::CONTAINER<int>::reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }
  {
    const std::CONTAINER<int> C;
    std::CONTAINER<int>::const_iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }
  {
    const std::CONTAINER<int> C;
    std::CONTAINER<int>::const_reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }

  return 0;
}
