// This checks that clang-scan-deps properly outputs named module dependencies 
// when using the the scanning output format 'experimental-full'.
//
// See PR #72304.
// UNSUPPORTED: target={{.*}}-aix{{.*}}
//
// RUN: rm -fr %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Check the separated dependency format.
// RUN: sed "s|DIR|%/t|g" %t/compile_commands.json.in > %t/compile_commands.json
// RUN: clang-scan-deps -format=experimental-full \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/M.cppm -o %t/M.o \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/M.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=experimental-full \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/Impl.cpp -o %t/Impl.o \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/Impl.cpp -DPREFIX=%/t
// RUN: clang-scan-deps -format=experimental-full \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/impl_part.cppm -o %t/impl_part.o \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/impl_part.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=experimental-full \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/interface_part.cppm -o %t/interface_part.o \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/interface_part.cppm -DPREFIX=%/t
// RUN: clang-scan-deps -format=experimental-full \
// RUN:   -- %clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/User.cpp -o %t/User.o \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/User.cpp -DPREFIX=%/t
//
// Check the combined dependency format.
// RUN: clang-scan-deps -compilation-database %t/compile_commands.json -format=experimental-full \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/Checks.cpp -DPREFIX=%/t
// RUN: clang-scan-deps --mode=preprocess-dependency-directives -compilation-database %t/compile_commands.json -format=experimental-full \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %t/Checks.cpp -DPREFIX=%/t

//--- compile_commands.json.in
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

// CHECK:       {
// CHECK:         "modules": []
// CHECK:         "translation-units": [
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M"
// CHECK-NEXT:            "named-module-deps": [
// CHECK-NEXT:              "M:interface_part",
// CHECK-NEXT:              "M:impl_part"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/M.o"
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/M.cppm"
// CHECK:               }
// CHECK:             ]
// CHECK:           }
// CHECK:         ]
// CHECK:       }

//--- Impl.cpp
module;
#include "header.mock"
module M;
void Hello() {
    std::cout << "Hello ";
}

// CHECK:       {
// CHECK:         "modules": []
// CHECK:         "translation-units": [
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK-NOT:             "named-module":
// CHECK:                 "named-module-deps": [
// CHECK-NEXT:              "M"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "[[PREFIX]]/Impl.o"
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/Impl.cpp"
// CHECK:               }
// CHECK:             ]
// CHECK:           }
// CHECK:         ]
// CHECK:       }

//--- impl_part.cppm
module;
#include "header.mock"
module M:impl_part;
import :interface_part;

std::string W = "World.";
void World() {
    std::cout << W << std::endl;
}

// CHECK:       {
// CHECK:         "modules": [],
// CHECK:         "translation-units": [
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M:impl_part"
// CHECK-NEXT:            "named-module-deps": [
// CHECK-NEXT:              "M:interface_part"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/impl_part.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/impl_part.cppm"
// CHECK:               }
// CHECK:             ]
// CHECK:           }
// CHECK:         ]
// CHECK:       }

//--- interface_part.cppm
export module M:interface_part;
export void World();

// CHECK:       {
// CHECK:         "modules": []
// CHECK:         "translation-units": [
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M:interface_part"
// CHECK-NOT:             "named-module-deps": []
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/interface_part.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/interface_part.cppm"
// CHECK:               }
// CHECK:             ]
// CHECK:           }
// CHECK:         ]
// CHECK:       }

//--- User.cpp
import M;
import third_party_module;
int main() {
    Hello();
    World();
    return 0;
}

// CHECK:       {
// CHECK-NEXT:    "modules": []
// CHECK-NEXT:    "translation-units": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "commands": [
// CHECK-NEXT:          {
// CHECK-NOT:             "named-module":
// CHECK:                 "named-module-deps": [
// CHECK-NEXT:              "M"
// CHECK-NEXT:              "third_party_module"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "[[PREFIX]]/User.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/User.cpp
// CHECK:               }
// CHECK:             ]
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK:       }

//--- header.mock
  
//--- Checks.cpp
// CHECK:       {
// CHECK:         "modules": []
// CHECK:         "translation-units": [
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M"
// CHECK-NEXT:            "named-module-deps": [
// CHECK-NEXT:              "M:interface_part",
// CHECK-NEXT:              "M:impl_part"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/M.o"
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/M.cppm"
// CHECK:               },
// CHECK:               {
// CHECK-NOT:             "named-module":
// CHECK:                 "named-module-deps": [
// CHECK-NEXT:              "M"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "[[PREFIX]]/Impl.o"
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/Impl.cpp"
// CHECK:               }
// CHECK:             ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M:impl_part"
// CHECK-NEXT:            "named-module-deps": [
// CHECK-NEXT:              "M:interface_part"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/impl_part.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/impl_part.cppm"
// CHECK:               }
// CHECK:             ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK:                 "named-module": "M:interface_part"
// CHECK-NOT:             "named-module-deps": []
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "{{.*}}/interface_part.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/interface_part.cppm"
// CHECK:               }
// CHECK:             ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK:             "commands": [
// CHECK-NEXT:          {
// CHECK-NOT:             "named-module":
// CHECK:                 "named-module-deps": [
// CHECK-NEXT:              "M"
// CHECK-NEXT:              "third_party_module"
// CHECK-NEXT:             ]
// CHECK:                 "command-line": [
// CHECK:                   "-o",
// CHECK-NEXT:              "[[PREFIX]]/User.o",
// CHECK:                 ]
// CHECK:                 "input-file": "[[PREFIX]]/User.cpp
// CHECK:               } 
// CHECK:             ]
// CHECK:           }
// CHECK:         ]
// CHECK:       }
