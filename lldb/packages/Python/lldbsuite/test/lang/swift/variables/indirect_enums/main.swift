// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
class C {
  var s = "Hello World"
}

class D : C {
  var x = 12
}

enum Simple
{
  case A,B,C,D
}

enum ADT
{
  case A(Int)
  case B(String)
}

enum GuineaPig
{
  indirect case StructType(Int)
  indirect case TupleType((Int,String))
  indirect case ClassType(C)
  indirect case ProtocolType(Any)
  indirect case CEnumType(Simple)
  indirect case ADTEnumType(ADT)
  indirect case Recursive(GuineaPig)
}

func main()
{
  var GP_StructType = GuineaPig.StructType(12)
  var GP_TupleType = GuineaPig.TupleType((12,"Hello World"))
  var GP_ClassType = GuineaPig.ClassType(D())
  var GP_ProtocolType_Struct = GuineaPig.ProtocolType(12)
  var GP_ProtocolType_Class = GuineaPig.ProtocolType(D())
  var GP_CEnumType = GuineaPig.CEnumType(Simple.B)
  var GP_ADTEnumType = GuineaPig.ADTEnumType(ADT.A(12))
  var GP_Recursive = GuineaPig.Recursive(GuineaPig.StructType(12))
  print("break here")
}

main()
