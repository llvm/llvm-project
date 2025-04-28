//===--- BuiltinsSPIRV.td - SPIRV Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def SPIRVDistance : Builtin {
  let Spellings = ["__builtin_spirv_distance"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVLength : Builtin {
  let Spellings = ["__builtin_spirv_length"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVReflect : Builtin {
  let Spellings = ["__builtin_spirv_reflect"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVSmoothStep : Builtin {
  let Spellings = ["__builtin_spirv_smoothstep"];
  let Attributes = [NoThrow, Const, CustomTypeChecking];
  let Prototype = "void(...)";
}

def SPIRVSmoothStep : Builtin {
  let Spellings = ["__builtin_spirv_faceforward"];
  let Attributes = [NoThrow, Const, CustomTypeChecking];
  let Prototype = "void(...)";
}

def SPIRVGetNumWrkgrpX : Builtin {
  let Spellings = ["__builtin_spirv_get_num_workgroups_x"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetNumWrkgrpY : Builtin {
  let Spellings = ["__builtin_spirv_get_num_workgroups_y"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetNumWrkgrpZ : Builtin {
  let Spellings = ["__builtin_spirv_get_num_workgroups_z"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpIdX : Builtin {
  let Spellings = ["__builtin_spirv_get_workgroup_id_x"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpIdY : Builtin {
  let Spellings = ["__builtin_spirv_get_workgroup_id_y"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpIdZ : Builtin {
  let Spellings = ["__builtin_spirv_get_workgroup_id_z"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpSizeX : Builtin {
  let Spellings = ["__builtin_spirv_workgroup_size_x"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpSizeY : Builtin {
  let Spellings = ["__builtin_spirv_workgroup_size_y"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkgrpSizeZ : Builtin {
  let Spellings = ["__builtin_spirv_workgroup_size_z"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkitemIdX : Builtin {
  let Spellings = ["__builtin_spirv_workitem_id_x"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkitemIdY : Builtin {
  let Spellings = ["__builtin_spirv_workitem_id_y"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVGetWrkitemIdZ : Builtin {
  let Spellings = ["__builtin_spirv_workitem_id_z"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVBallot : Builtin {
  let Spellings = ["__builtin_spirv_ballot"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVSyncThreads : Builtin {
  let Spellings = ["__builtin_spirv_sync_threads"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}
