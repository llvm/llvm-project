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

func main() -> Int {
    var int8_minus_two  = Int8(-2)
    var int8_plus_two   = Int8(2)
    var int16_minus_two = Int16(-2)
    var int16_plus_two  = Int16(2)
    var int32_minus_two = Int32(-2)
    var int32_plus_two  = Int32(2)
    var int64_minus_two = -2
    var int64_plus_two  = 2
    var int_minus_two   = -2
    var int_plus_two    = 2

    var uint8_plus_two   = UInt8(2)
    var uint16_plus_two  = UInt16(2)
    var uint32_plus_two  = UInt32(2)
    var uint64_plus_two  = UInt64(2)
    
    var float32 = Float32(1.25)
    var float64 = 2.5
#if !os(iOS)
    var float80 = Float80(1.0625)
#endif
    var float = Float(3.75)
    
    var hello = "hello"
    var world = "world"
    var hello_world = hello + " " + world

    var uint8_max = UInt8.max
    var uint16_max = UInt16.max
    var uint32_max = UInt32.max
    var uint64_max = UInt64.max
    var uint_max = UInt.max

    print("stop here") // Set breakpoint here
    return 0
}

main()
