// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: sed 's,^//.*,//,' %s > %t/absolute-fixed.cpp
// RUN: sed 's,^//.*,//,' %s > %t/absolute-json.cpp
// RUN: sed 's,^//.*,//,' %s > %t/relative-fixed.cpp
// RUN: sed 's,^//.*,//,' %s > %t/relative-json.cpp
//
// RUN: clang-check %t/absolute-fixed.cpp -fixit -- 2>&1 | FileCheck %s
//
// RUN: echo "[{ \"directory\":\"%/t\", \"command\":\"/path/to/clang -c %/t/absolute-json.cpp\", \"file\": \"%/t/absolute-json.cpp\" }]" > %t/compile_commands.json
// RUN: clang-check %t/absolute-json.cpp -fixit 2>&1 | FileCheck %s
//
// RUN: cd %t
// RUN: clang-check relative-fixed.cpp -fixit -- 2>&1 | FileCheck %s
//
// RUN: echo "[{ \"directory\": \"%/t\", \"command\": \"/path/to/clang -c relative-json.cpp\", \"file\": \"relative-json.cpp\" }]" > %t/compile_commands.json
// RUN: clang-check relative-json.cpp -fixit 2>&1 | FileCheck %s
typedef int T
// CHECK: .cpp:[[@LINE-1]]:14: error: expected ';' after top level declarator
// CHECK: .cpp:[[@LINE-2]]:14: note: FIX-IT applied suggested code changes
