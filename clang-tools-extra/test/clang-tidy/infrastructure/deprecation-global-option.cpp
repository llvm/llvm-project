// RUN: clang-tidy %s --config="{CheckOptions:{StrictMode: true}}" -checks="-*,modernize-use-std-format" | FileCheck %s 

// CHECK: warning: deprecation global option 'StrictMode', please use 'modernize-use-std-format.StrictMode'. [clang-tidy-config]
