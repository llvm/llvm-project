// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++20 -emit-llvm -o - %t/mongod.cpp | FileCheck %s --check-prefix=MONGOD
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++20 -emit-llvm -o - %t/mongos.cpp | FileCheck %s --check-prefix=MONGOS

// Reduced from MongoDB's metric registration: two translation units include
// shared inline options, then register a metric during ordinary namespace-scope
// dynamic initialization. The options definition precedes both registrations,
// so its partially ordered initialization must precede their initialization.
// Unlike inline static data members, this already has the correct order in IR.
// ms-inline-variable-registration-init-order-coff.cpp checks the object file.

// MONGOD: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }]
// MONGOD-SAME: { i32 65535, ptr @"??__Eoptions@@YAXXZ", ptr @"?options@@3UOptions@@B" }
// MONGOD-SAME: { i32 65535, ptr @_GLOBAL__sub_I_mongod.cpp, ptr null }
// MONGOD-LABEL: define internal void @_GLOBAL__sub_I_mongod.cpp()
// MONGOD: call void @"??__Emongod@@YAXXZ"()

// MONGOS: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }]
// MONGOS-SAME: { i32 65535, ptr @"??__Eoptions@@YAXXZ", ptr @"?options@@3UOptions@@B" }
// MONGOS-SAME: { i32 65535, ptr @_GLOBAL__sub_I_mongos.cpp, ptr null }
// MONGOS-LABEL: define internal void @_GLOBAL__sub_I_mongos.cpp()
// MONGOS: call void @"??__Emongos@@YAXXZ"()

//--- options.h
struct Options {
  int serverStatusPath;
};

// Force dynamic initialization without depending on a standard library.
int makeServerStatusPath();
inline const Options options = [] { return Options{makeServerStatusPath()}; }();

struct Registration {
  int observedPath;
  explicit Registration(const Options &opts)
      : observedPath(opts.serverStatusPath) {}
};

//--- mongod.cpp
#include "options.h"
Registration mongod(options);

//--- mongos.cpp
#include "options.h"
Registration mongos(options);
