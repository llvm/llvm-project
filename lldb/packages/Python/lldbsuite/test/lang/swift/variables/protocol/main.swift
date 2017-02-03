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
protocol PointUtils {
  func sumOfCoordinates () -> Float
}

struct Point2D : PointUtils {
    var x : Float
    var y : Float
    init (_ _x : Float, _ _y : Float) {
        x = _x;
        y = _y;
    }
    
    func sumOfCoordinates () -> Float {
        var sum = x + y
        return sum
    }
    
}

class Point3D : PointUtils {
    var x : Float
    var y : Float
    var z : Float

    init (_ _x : Float, _ _y : Float, _ _z : Float) {
        x = _x;
        y = _y;
        z = _z
    }
    
    func sumOfCoordinates () -> Float {
        var sum = x + y + z
        return sum
    }
    
}

func main() -> Int {
    
    var loc2d : protocol<PointUtils> = Point2D(1.25, 2.5)
    var loc3d : protocol<PointUtils> = Point3D(1.25, 2.5, 1.25)
    var sum2d = loc2d.sumOfCoordinates()
    var sum3d = loc3d.sumOfCoordinates()
    print("hello \(sum2d) \(sum3d)") // Set breakpoint here
    return 0
}

main()

