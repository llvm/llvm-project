// Test case from https://github.com/llvm/llvm-project/issues/59999
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Module-Reflector.cppm \
// RUN:     -emit-module-interface -o %t/Module-Reflector.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Module.cppm \
// RUN:     -emit-module-interface -o %t/Module.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Object-Reflector.cppm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -emit-module-interface \
// RUN:     -o %t/Object-Reflector.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Object.cppm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -emit-module-interface -o %t/Object.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/World.cppm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -fmodule-file=Object=%t/Object.pcm \
// RUN:     -o %t/World.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/World.pcm \
// RUN:     -fmodule-file=Module=%t/Module.pcm -fmodule-file=Object=%t/Object.pcm \
// RUN:     -S -emit-llvm -o - | FileCheck %t/World.cppm

//--- Module-Reflector.cppm
export module Module:Reflector;

export template <class ObjectType> bool ModuleRegister() { return true; };

//--- Module.cppm
export module Module;

export import :Reflector;

export struct ModuleEntry {
  static const bool bRegistered;
};

const bool ModuleEntry::bRegistered = ModuleRegister<ModuleEntry>();

//--- Object-Reflector.cppm
export module Object:Reflector;

export template <class ObjectType> bool ObjectRegister() { return true; };

//--- Object.cppm
export module Object;

import Module;

export import :Reflector;

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

//--- World.cppm

export module World;

import Object;

export const bool NWorldRegistered = ModuleRegister<long>();

// Check that the LLVM IR is generated correctly instead of crashing.
// CHECK: define{{.*}}@_ZGIW6Object
