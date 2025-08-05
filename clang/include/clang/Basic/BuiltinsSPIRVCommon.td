//===- BuiltinsSPIRVCommon.td - SPIRV Builtin function database -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsSPIRVBase.td"

def num_workgroups : SPIRVBuiltin<"size_t(int)", [NoThrow, Const]>;
def workgroup_size : SPIRVBuiltin<"size_t(int)", [NoThrow, Const]>;
def workgroup_id : SPIRVBuiltin<"size_t(int)", [NoThrow, Const]>;
def local_invocation_id : SPIRVBuiltin<"size_t(int)", [NoThrow, Const]>;
def global_invocation_id : SPIRVBuiltin<"size_t(int)", [NoThrow, Const]>;
def subgroup_size : SPIRVBuiltin<"uint32_t()", [NoThrow, Const]>;
def num_subgroups : SPIRVBuiltin<"uint32_t()", [NoThrow, Const]>;
def subgroup_id : SPIRVBuiltin<"uint32_t()", [NoThrow, Const]>;
def subgroup_local_invocation_id : SPIRVBuiltin<"uint32_t()", [NoThrow, Const]>;

def distance : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def length : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def smoothstep : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
