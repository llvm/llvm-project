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
enum One {
	case A
	case B
	case C
	case D
}

enum Two {
	case A
	case B(String)
	case C(Int)
	case D
}

enum Three {
	case A(String)
	case B(Int)
	case C(String)
	case D(Bool)
}

enum Four {
	case A(String)
	case B(String)
	case C(String)
	case D(String)
}

struct ContainerOfEnums {
  var one1: One
  var one2: One
}

func main() {
	var ona = One.A
	var onb = One.B
	var onc = One.C
	var ond = One.D

	var twa = Two.A
	var twb = Two.B("hello world")
	var twc = Two.C(12)
	var twd = Two.D

	var tha = Three.A("hello world")
	var thb = Three.B(24)
	var thc = Three.C("this is me")
	var thd = Three.D(true)

	var foa = Four.A("hello world")
	var fob = Four.B("this is me")
	var foc = Four.C("life should be")
	var fod = Four.D("fun for everyone")
  
  var ContainerOfEnums_Some: ContainerOfEnums? = ContainerOfEnums(one1: .A, one2: .A)
  var ContainerOfEnums_Nil: ContainerOfEnums? = nil

	print("hello world") // Set breakpoint here
}

main()
