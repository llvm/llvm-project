// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --doxygen --executor=standalone %S/../Inputs/templates.cpp -output=%t/docs --format=md
// RUN: FileCheck %s --check-prefix=MD < %t/docs/md/GlobalNamespace/index.md

// MD: # Global Namespace
// MD: ## Functions

// MD: ### ParamPackFunction
// MD: *void ParamPackFunction(T... args)*

// MD: ### function
// MD: *void function(T x)*
// MD: *Defined at {{.*}}templates.cpp#3*

// MD: ### function
// MD: *void function(bool x)*
// MD: *Defined at {{.*}}templates.cpp#8*

// MD: ### func_with_tuple_param
// MD: *tuple<int, int, bool> func_with_tuple_param(tuple<int, int, bool> t)*
// MD: *Defined at {{.*}}templates.cpp#18*
// MD:  A function with a tuple parameter
// MD: **t** The input to func_with_tuple_param
