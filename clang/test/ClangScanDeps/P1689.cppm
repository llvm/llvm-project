// The slash direction in linux and windows are different.
// UNSUPPORTED: system-windows
//
// RUN: rm -fr %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: sed "s|DIR|%/t|g" %t/P1689.json.in > %t/P1689.json
// RUN: clang-scan-deps -compilation-database %t/P1689.json -format=p1689 | FileCheck %t/Checks.cpp -DPREFIX=%/t
// RUN: clang-scan-deps --mode=preprocess-dependency-directives -compilation-database %t/P1689.json -format=p1689 | FileCheck %t/Checks.cpp -DPREFIX=%/t
//
// Check the separated dependency format. This is required by CMake for the case
// that we have non-exist files in a fresh build and potentially out-of-date after that.
// So the build system need to wrtie a compilation database just for scanning purposes,
// which is not so good. So here is the per file mode for P1689.
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/M.cppm -o %t/M.o \
// RUN:   | FileCheck %t/M.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/Impl.cpp -o %t/Impl.o \
// RUN:   | FileCheck %t/Impl.cpp -DPREFIX=%/t
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/impl_part.cppm -o %t/impl_part.o \
// RUN:   | FileCheck %t/impl_part.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/interface_part.cppm -o %t/interface_part.o \
// RUN:   | FileCheck %t/interface_part.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/User.cpp -o %t/User.o \
// RUN:   | FileCheck %t/User.cpp -DPREFIX=%/t
//
// Check we can generate the make-style dependencies as expected.
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/impl_part.cppm -o %t/impl_part.o \
// RUN:      -MT %t/impl_part.o.ddi -MD -MF %t/impl_part.dep
// RUN:   cat %t/impl_part.dep | FileCheck %t/impl_part.cppm -DPREFIX=%/t --check-prefix=CHECK-MAKE
//
// Check that we can generate multiple make-style dependency information with compilation database.
// RUN: cat %t/P1689.dep | FileCheck %t/Checks.cpp -DPREFIX=%/t --check-prefix=CHECK-MAKE
//
// Check that we can mix the use of -format=p1689 and -fmodules.
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %clang++ -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -serialize-diagnostics -c %t/impl_part.cppm -o %t/impl_part.o \
// RUN:   | FileCheck %t/impl_part.cppm -DPREFIX=%/t

//--- P1689.json.in
[
{
  "directory": "DIR",
  "command": "clang++ -std=c++20 DIR/M.cppm -c -o DIR/M.o -MT DIR/M.o.ddi -MD -MF DIR/P1689.dep",
  "file": "DIR/M.cppm",
  "output": "DIR/M.o"
},
{
  "directory": "DIR",
  "command": "clang++ -std=c++20 DIR/Impl.cpp -c -o DIR/Impl.o -MT DIR/Impl.o.ddi -MD -MF DIR/P1689.dep",
  "file": "DIR/Impl.cpp",
  "output": "DIR/Impl.o"
},
{
  "directory": "DIR",
  "command": "clang++ -std=c++20 DIR/impl_part.cppm -c -o DIR/impl_part.o -MT DIR/impl_part.o.ddi -MD -MF DIR/P1689.dep",
  "file": "DIR/impl_part.cppm",
  "output": "DIR/impl_part.o"
},
{
  "directory": "DIR",
  "command": "clang++ -std=c++20 DIR/interface_part.cppm -c -o DIR/interface_part.o -MT DIR/interface_part.o.ddi -MD -MF DIR/P1689.dep",
  "file": "DIR/interface_part.cppm",
  "output": "DIR/interface_part.o"
},
{
  "directory": "DIR",
  "command": "clang++ -std=c++20 DIR/User.cpp -c -o DIR/User.o -MT DIR/User.o.ddi -MD -MF DIR/P1689.dep",
  "file": "DIR/User.cpp",
  "output": "DIR/User.o"
}
]

//--- M.cppm
export module M;
export import :interface_part;
import :impl_part;
export void Hello();

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/M.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": true,
// CHECK-NEXT:           "logical-name": "M",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/M.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:interface_part"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:impl_part"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

//--- Impl.cpp
module;
#include "header.mock"
module M;
void Hello() {
    std::cout << "Hello ";
}

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/Impl.o",
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

//--- impl_part.cppm
module;
#include "header.mock"
module M:impl_part;
import :interface_part;

std::string W = "World.";
void World() {
    std::cout << W << std::endl;
}

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/impl_part.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": false,
// CHECK-NEXT:           "logical-name": "M:impl_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/impl_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:interface_part"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

// CHECK-MAKE: [[PREFIX]]/impl_part.o.ddi:
// CHECK-MAKE:   [[PREFIX]]/impl_part.cppm
// CHECK-MAKE:   [[PREFIX]]/header.mock

//--- interface_part.cppm
export module M:interface_part;
export void World();

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/interface_part.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": true,
// CHECK-NEXT:           "logical-name": "M:interface_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/interface_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

//--- User.cpp
import M;
import third_party_module;
int main() {
    Hello();
    World();
    return 0;
}

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/User.o",
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "third_party_module"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

//--- Checks.cpp
// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/Impl.o",
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/M.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/M.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": true,
// CHECK-NEXT:           "logical-name": "M",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/M.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:interface_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/interface_part.cppm"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:impl_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/impl_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/User.o",
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/M.cppm"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "third_party_module"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/impl_part.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": false,
// CHECK-NEXT:           "logical-name": "M:impl_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/impl_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "M:interface_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/interface_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/interface_part.o",
// CHECK-NEXT:       "provides": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "is-interface": true,
// CHECK-NEXT:           "logical-name": "M:interface_part",
// CHECK-NEXT:           "source-path": "[[PREFIX]]/interface_part.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

// CHECK-MAKE-DAG: [[PREFIX]]/impl_part.o.ddi: \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/impl_part.cppm \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/header.mock
// CHECK-MAKE-DAG: [[PREFIX]]/interface_part.o.ddi: \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/interface_part.cppm
// CHECK-MAKE-DAG: [[PREFIX]]/M.o.ddi: \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/M.cppm
// CHECK-MAKE-DAG: [[PREFIX]]/User.o.ddi: \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/User.cpp
// CHECK-MAKE-DAG: [[PREFIX]]/Impl.o.ddi: \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/Impl.cpp \
// CHECK-MAKE-DAG-NEXT:   [[PREFIX]]/header.mock

//--- module.modulemap
module Mock { header "header.mock" }

//--- header.mock
