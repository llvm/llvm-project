// Test case from https://github.com/llvm/llvm-project/issues/59999
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Module.cppm \
// RUN:     -emit-module-interface -o %t/Module.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Object.cppm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -emit-module-interface -o %t/Object.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Object.pcm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -S -emit-llvm -o - | FileCheck %t/Object.cppm

//--- Module.cppm
export module Module;

export template <class ObjectType> bool ModuleRegister() { return true; };

export struct ModuleEntry {
  static const bool bRegistered;
};

const bool ModuleEntry::bRegistered = ModuleRegister<ModuleEntry>();

//--- Object.cppm
export module Object;

import Module;

export template <class ObjectType> bool ObjectRegister() { return true; }
export struct NObject {
  static const bool bRegistered;
};
export struct ObjectModuleEntry {
  static const bool bRegistered;
};

// This function is also required for crash
const bool NObject::bRegistered = ObjectRegister<NObject>();
// One another function, that helps clang crash
const bool ObjectModuleEntry::bRegistered = ModuleRegister<ObjectModuleEntry>();

// Check that the LLVM IR is generated correctly instead of crashing.
// CHECK: define{{.*}}@_ZGIW6Object
