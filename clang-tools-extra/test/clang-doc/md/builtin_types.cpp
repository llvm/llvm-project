// RUN: rm -rf %t
// RUN: mkdir -p %t/yaml %t/md %t/md_mustache
// RUN: clang-doc --doxygen --executor=standalone %S/../Inputs/builtin_types.cpp -output=%t/md --format=md
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md  --check-prefix=MD
// RUN: clang-doc --doxygen --executor=standalone %S/../Inputs/builtin_types.cpp -output=%t/md_mustache --format=md_mustache
// RUN: FileCheck %s < %t/md_mustache/md/GlobalNamespace/index.md  --check-prefix=MD-MUSTACHE

// MD: # Global Namespace
// MD: ## Functions

// MD-MUSTACHE: # Global Namespace
// MD-MUSTACHE: ## Functions

// MD: ### b
// MD: *bool b()*

// MD-MUSTACHE: ### b
// MD-MUSTACHE: *bool b()*

// MD: ### c
// MD: *char c()*

// MD-MUSTACHE: ### c
// MD-MUSTACHE: *char c()*

// MD: ### d
// MD: *double d()*

// MD-MUSTACHE: ### d
// MD-MUSTACHE: *double d()*

// MD: ### f
// MD: *float f()*

// MD-MUSTACHE: ### f
// MD-MUSTACHE: *float f()*

// MD: ### i
// MD: *int i()*

// MD-MUSTACHE: ### i
// MD-MUSTACHE: *int i()*

// MD: ### l
// MD: *long l()*

// MD-MUSTACHE: ### l
// MD-MUSTACHE: *long l()*

// MD: ### ll
// MD: *long long ll()*

// MD-MUSTACHE: ### ll
// MD-MUSTACHE: *long long ll()*

// MD: ### s
// MD: *short s()*

// MD-MUSTACHE: ### s
// MD-MUSTACHE: *short s()*
