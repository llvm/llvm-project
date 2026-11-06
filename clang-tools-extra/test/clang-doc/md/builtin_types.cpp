// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --doxygen --executor=standalone %S/../Inputs/builtin_types.cpp --output=%t --format=md
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD

// MD: # Global Namespace
// MD: ## Functions

// MD: ### b
// MD: *bool b()*

// MD: ### c
// MD: *char c()*

// MD: ### d
// MD: *double d()*

// MD: ### f
// MD: *float f()*

// MD: ### i
// MD: *int i()*

// MD: ### l
// MD: *long l()*

// MD: ### ll
// MD: *long long ll()*

// MD: ### s
// MD: *short s()*
