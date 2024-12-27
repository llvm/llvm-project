// RUN: clang-tidy %s --config="{CheckOptions:{StrictMode: true}}" -checks="-*,modernize-use-std-format" | FileCheck %s 

// CHECK: warning: global option 'StrictMode' is deprecated, please use 'modernize-use-std-format.StrictMode' instead. [clang-tidy-config]
