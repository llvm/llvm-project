//===--- Opcodes.td - Opcode defitions for the constexpr VM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper file used to generate opcodes, the interpreter and the disassembler.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Types evaluated by the interpreter.
//===----------------------------------------------------------------------===//

class Type;
def Bool : Type;
def Sint8 : Type;
def Uint8 : Type;
def Sint16 : Type;
def Uint16 : Type;
def Sint32 : Type;
def Uint32 : Type;
def Sint64 : Type;
def Uint64 : Type;
def Float : Type;
def Ptr : Type;
def FnPtr : Type;

//===----------------------------------------------------------------------===//
// Types transferred to the interpreter.
//===----------------------------------------------------------------------===//

class ArgType { string Name = ?; }
def ArgSint8 : ArgType { let Name = "int8_t"; }
def ArgUint8 : ArgType { let Name = "uint8_t"; }
def ArgSint16 : ArgType { let Name = "int16_t"; }
def ArgUint16 : ArgType { let Name = "uint16_t"; }
def ArgSint32 : ArgType { let Name = "int32_t"; }
def ArgUint32 : ArgType { let Name = "uint32_t"; }
def ArgSint64 : ArgType { let Name = "int64_t"; }
def ArgUint64 : ArgType { let Name = "uint64_t"; }
def ArgFloat : ArgType { let Name = "Floating"; }
def ArgBool : ArgType { let Name = "bool"; }

def ArgFunction : ArgType { let Name = "const Function *"; }
def ArgRecordDecl : ArgType { let Name = "const RecordDecl *"; }
def ArgRecordField : ArgType { let Name = "const Record::Field *"; }
def ArgFltSemantics : ArgType { let Name = "const llvm::fltSemantics *"; }
def ArgRoundingMode : ArgType { let Name = "llvm::RoundingMode"; }
def ArgLETD: ArgType { let Name = "const LifetimeExtendedTemporaryDecl *"; }

//===----------------------------------------------------------------------===//
// Classes of types instructions operate on.
//===----------------------------------------------------------------------===//

class TypeClass {
  list<Type> Types;
}

def IntegerTypeClass : TypeClass {
  let Types = [Sint8, Uint8, Sint16, Uint16, Sint32,
               Uint32, Sint64, Uint64];
}

def NumberTypeClass : TypeClass {
  let Types = !listconcat(IntegerTypeClass.Types, [Float]);
}

def FloatTypeClass : TypeClass {
  let Types = [Float];
}

def AluTypeClass : TypeClass {
  let Types = !listconcat(IntegerTypeClass.Types, [Bool]);
}

def PtrTypeClass : TypeClass {
  let Types = [Ptr, FnPtr];
}

def BoolTypeClass : TypeClass {
  let Types = [Bool];
}

def NonPtrTypeClass : TypeClass {
  let Types = !listconcat(IntegerTypeClass.Types, [Bool], [Float]);
}

def AllTypeClass : TypeClass {
  let Types = !listconcat(AluTypeClass.Types, PtrTypeClass.Types, FloatTypeClass.Types);
}

def ComparableTypeClass : TypeClass {
  let Types = !listconcat(AluTypeClass.Types, [Ptr], [Float]);
}

class SingletonTypeClass<Type Ty> : TypeClass {
  let Types = [Ty];
}

//===----------------------------------------------------------------------===//
// Record describing all opcodes.
//===----------------------------------------------------------------------===//

class Opcode {
  list<TypeClass> Types = [];
  list<ArgType> Args = [];
  string Name = "";
  bit CanReturn = 0;
  bit ChangesPC = 0;
  bit HasCustomLink = 0;
  bit HasCustomEval = 0;
  bit HasGroup = 0;
}

class AluOpcode : Opcode {
  let Types = [AluTypeClass];
  let HasGroup = 1;
}

class FloatOpcode : Opcode {
  let Types = [];
  let Args = [ArgRoundingMode];
}

class IntegerOpcode : Opcode {
  let Types = [IntegerTypeClass];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Jump opcodes
//===----------------------------------------------------------------------===//

class JumpOpcode : Opcode {
  let Args = [ArgSint32];
  let ChangesPC = 1;
  let HasCustomEval = 1;
}

// [] -> []
def Jmp : JumpOpcode;
// [Bool] -> [], jumps if true.
def Jt : JumpOpcode;
// [Bool] -> [], jumps if false.
def Jf : JumpOpcode;

//===----------------------------------------------------------------------===//
// Returns
//===----------------------------------------------------------------------===//

// [Value] -> []
def Ret : Opcode {
  let Types = [AllTypeClass];
  let ChangesPC = 1;
  let CanReturn = 1;
  let HasGroup = 1;
  let HasCustomEval = 1;
}
// [] -> []
def RetVoid : Opcode {
  let CanReturn = 1;
  let ChangesPC = 1;
  let HasCustomEval = 1;
}
// [Value] -> []
def RetValue : Opcode {
  let CanReturn = 1;
  let ChangesPC = 1;
  let HasCustomEval = 1;
}
// [] -> EXIT
def NoRet : Opcode {}


def Call : Opcode {
  let Args = [ArgFunction];
  let Types = [];
  let ChangesPC = 1;
}

def CallBI : Opcode {
  let Args = [ArgFunction];
  let Types = [];
  let ChangesPC = 1;
}

def CallPtr : Opcode {
  let Args = [];
  let Types = [];
  let ChangesPC = 1;
}

//===----------------------------------------------------------------------===//
// Frame management
//===----------------------------------------------------------------------===//

// [] -> []
def Destroy : Opcode {
  let Args = [ArgUint32];
  let HasCustomEval = 1;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

class ConstOpcode<Type Ty, ArgType ArgTy> : Opcode {
  let Types = [SingletonTypeClass<Ty>];
  let Args = [ArgTy];
  let Name = "Const";
}

// [] -> [Integer]
def ConstSint8 : ConstOpcode<Sint8, ArgSint8>;
def ConstUint8 : ConstOpcode<Uint8, ArgUint8>;
def ConstSint16 : ConstOpcode<Sint16, ArgSint16>;
def ConstUint16 : ConstOpcode<Uint16, ArgUint16>;
def ConstSint32 : ConstOpcode<Sint32, ArgSint32>;
def ConstUint32 : ConstOpcode<Uint32, ArgUint32>;
def ConstSint64 : ConstOpcode<Sint64, ArgSint64>;
def ConstUint64 : ConstOpcode<Uint64, ArgUint64>;
def ConstFloat : ConstOpcode<Float, ArgFloat>;
def ConstBool : ConstOpcode<Bool, ArgBool>;

// [] -> [Integer]
def Zero : Opcode {
  let Types = [AluTypeClass];
  let HasGroup = 1;
}

// [] -> [Pointer]
def Null : Opcode {
  let Types = [PtrTypeClass];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Pointer generation
//===----------------------------------------------------------------------===//

// [] -> [Pointer]
def GetPtrLocal : Opcode {
  // Offset of local.
  let Args = [ArgUint32];
  bit HasCustomEval = 1;
}
// [] -> [Pointer]
def GetPtrParam : Opcode {
  // Offset of parameter.
  let Args = [ArgUint32];
}
// [] -> [Pointer]
def GetPtrGlobal : Opcode {
  // Index of global.
  let Args = [ArgUint32];
}
// [Pointer] -> [Pointer]
def GetPtrField : Opcode {
  // Offset of field.
  let Args = [ArgUint32];
}
// [Pointer] -> [Pointer]
def GetPtrActiveField : Opcode {
  // Offset of field.
  let Args = [ArgUint32];
}
// [] -> [Pointer]
def GetPtrActiveThisField : Opcode {
  // Offset of field.
  let Args = [ArgUint32];
}
// [] -> [Pointer]
def GetPtrThisField : Opcode {
  // Offset of field.
  let Args = [ArgUint32];
}
// [Pointer] -> [Pointer]
def GetPtrBase : Opcode {
  // Offset of field, which is a base.
  let Args = [ArgUint32];
}
// [Pointer] -> [Pointer]
def GetPtrBasePop : Opcode {
  // Offset of field, which is a base.
  let Args = [ArgUint32];
}

// [Pointer] -> [Pointer]
def GetPtrVirtBase : Opcode {
  // RecordDecl of base class.
  let Args = [ArgRecordDecl];
}
// [] -> [Pointer]
def GetPtrThisBase : Opcode {
  // Offset of field, which is a base.
  let Args = [ArgUint32];
}
// [] -> [Pointer]
def GetPtrThisVirtBase : Opcode {
  // RecordDecl of base class.
  let Args = [ArgRecordDecl];
}
// [] -> [Pointer]
def This : Opcode;

// [] -> [Pointer]
def RVOPtr : Opcode;

// [Pointer] -> [Pointer]
def NarrowPtr : Opcode;
// [Pointer] -> [Pointer]
def ExpandPtr : Opcode;
// [Pointer, Offset] -> [Pointer]
def ArrayElemPtr : AluOpcode;
def ArrayElemPtrPop : AluOpcode;

//===----------------------------------------------------------------------===//
// Direct field accessors
//===----------------------------------------------------------------------===//

class AccessOpcode : Opcode {
  let Types = [AllTypeClass];
  let Args = [ArgUint32];
  let HasGroup = 1;
}

class BitFieldOpcode : Opcode {
  let Types = [AluTypeClass];
  let Args = [ArgRecordField];
  let HasGroup = 1;
}

// [] -> [Pointer]
def GetLocal : AccessOpcode { let HasCustomEval = 1; }
// [] -> [Pointer]
def SetLocal : AccessOpcode { let HasCustomEval = 1; }

def CheckGlobalCtor : Opcode {}

// [] -> [Value]
def GetGlobal : AccessOpcode;
// [Value] -> []
def InitGlobal : AccessOpcode;
// [Value] -> []
def InitGlobalTemp : AccessOpcode {
  let Args = [ArgUint32, ArgLETD];
}
// [Value] -> []
def SetGlobal : AccessOpcode;

// [] -> [Value]
def GetParam : AccessOpcode;
// [Value] -> []
def SetParam : AccessOpcode;

// [Pointer] -> [Pointer, Value]
def GetField : AccessOpcode;
// [Pointer] -> [Value]
def GetFieldPop : AccessOpcode;
// [] -> [Value]
def GetThisField : AccessOpcode;

// [Pointer, Value] -> [Pointer]
def SetField : AccessOpcode;
// [Value] -> []
def SetThisField : AccessOpcode;

// [Value] -> []
def InitThisField : AccessOpcode;
// [Value] -> []
def InitThisFieldActive : AccessOpcode;
// [Value] -> []
def InitThisBitField : BitFieldOpcode;
// [Pointer, Value] -> []
def InitField : AccessOpcode;
// [Pointer, Value] -> []
def InitBitField : BitFieldOpcode;
// [Pointer, Value] -> []
def InitFieldActive : AccessOpcode;

//===----------------------------------------------------------------------===//
// Pointer access
//===----------------------------------------------------------------------===//

class LoadOpcode : Opcode {
  let Types = [AllTypeClass];
  let HasGroup = 1;
}

// [Pointer] -> [Pointer, Value]
def Load : LoadOpcode {}
// [Pointer] -> [Value]
def LoadPop : LoadOpcode {}

class StoreOpcode : Opcode {
  let Types = [AllTypeClass];
  let HasGroup = 1;
}

class StoreBitFieldOpcode : Opcode {
  let Types = [AluTypeClass];
  let HasGroup = 1;
}

// [Pointer, Value] -> [Pointer]
def Store : StoreOpcode {}
// [Pointer, Value] -> []
def StorePop : StoreOpcode {}

// [Pointer, Value] -> [Pointer]
def StoreBitField : StoreBitFieldOpcode {}
// [Pointer, Value] -> []
def StoreBitFieldPop : StoreBitFieldOpcode {}

// [Pointer, Value] -> []
def InitPop : StoreOpcode {}
// [Pointer, Value] -> [Pointer]
def InitElem : Opcode {
  let Types = [AllTypeClass];
  let Args = [ArgUint32];
  let HasGroup = 1;
}
// [Pointer, Value] -> []
def InitElemPop : Opcode {
  let Types = [AllTypeClass];
  let Args = [ArgUint32];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Pointer arithmetic.
//===----------------------------------------------------------------------===//

// [Pointer, Integral] -> [Pointer]
def AddOffset : AluOpcode;
// [Pointer, Integral] -> [Pointer]
def SubOffset : AluOpcode;

// [Pointer, Pointer] -> [Integral]
def SubPtr : Opcode {
  let Types = [IntegerTypeClass];
  let HasGroup = 1;
}

// [Pointer] -> [Pointer]
def IncPtr : Opcode {
  let HasGroup = 0;
}
// [Pointer] -> [Pointer]
def DecPtr : Opcode {
  let HasGroup = 0;
}

//===----------------------------------------------------------------------===//
// Function pointers.
//===----------------------------------------------------------------------===//
def GetFnPtr : Opcode {
  let Args = [ArgFunction];
}


//===----------------------------------------------------------------------===//
// Binary operators.
//===----------------------------------------------------------------------===//

// [Real, Real] -> [Real]
def Add  : AluOpcode;
def Addf : FloatOpcode;
def Sub  : AluOpcode;
def Subf : FloatOpcode;
def Mul  : AluOpcode;
def Mulf : FloatOpcode;
def Rem  : IntegerOpcode;
def Div  : IntegerOpcode;
def Divf : FloatOpcode;

def BitAnd : IntegerOpcode;
def BitOr : IntegerOpcode;
def BitXor : IntegerOpcode;

def Shl : Opcode {
  let Types = [IntegerTypeClass, IntegerTypeClass];
  let HasGroup = 1;
}

def Shr : Opcode {
  let Types = [IntegerTypeClass, IntegerTypeClass];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Unary operators.
//===----------------------------------------------------------------------===//

// [Real] -> [Real]
def Inv: Opcode {
  let Types = [BoolTypeClass];
  let HasGroup = 1;
}

def Inc: IntegerOpcode;
def IncPop : IntegerOpcode;
def Dec: IntegerOpcode;
def DecPop: IntegerOpcode;

// [Real] -> [Real]
def Neg: Opcode {
  let Types = [NonPtrTypeClass];
  let HasGroup = 1;
}

// [Real] -> [Real]
def Comp: Opcode {
  let Types = [IntegerTypeClass];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Cast, CastFP.
//===----------------------------------------------------------------------===//

def FromCastTypeClass : TypeClass {
  let Types = [Uint8, Sint8, Uint16, Sint16, Uint32, Sint32, Uint64, Sint64, Bool];
}

def ToCastTypeClass : TypeClass {
  let Types = [Uint8, Sint8, Uint16, Sint16, Uint32, Sint32, Uint64, Sint64, Bool];
}

def Cast: Opcode {
  let Types = [FromCastTypeClass, ToCastTypeClass];
  let HasGroup = 1;
}

def CastFP : Opcode {
  let Types = [];
  let Args = [ArgFltSemantics, ArgRoundingMode];
}

// Cast an integer to a floating type
def CastIntegralFloating : Opcode {
  let Types = [AluTypeClass];
  let Args = [ArgFltSemantics, ArgRoundingMode];
  let HasGroup = 1;
}

// Cast a floating to an integer type
def CastFloatingIntegral : Opcode {
  let Types = [AluTypeClass];
  let Args = [];
  let HasGroup = 1;
}

//===----------------------------------------------------------------------===//
// Comparison opcodes.
//===----------------------------------------------------------------------===//

class EqualityOpcode : Opcode {
  let Types = [AllTypeClass];
  let HasGroup = 1;
}

def EQ : EqualityOpcode;
def NE : EqualityOpcode;

class ComparisonOpcode : Opcode {
  let Types = [ComparableTypeClass];
  let HasGroup = 1;
}

def LT : ComparisonOpcode;
def LE : ComparisonOpcode;
def GT : ComparisonOpcode;
def GE : ComparisonOpcode;

//===----------------------------------------------------------------------===//
// Stack management.
//===----------------------------------------------------------------------===//

// [Value] -> []
def Pop : Opcode {
  let Types = [AllTypeClass];
  let HasGroup = 1;
}

// [Value] -> [Value, Value]
def Dup : Opcode {
  let Types = [AllTypeClass];
  let HasGroup = 1;
}
