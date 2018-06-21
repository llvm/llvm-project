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

class PointSuperclass {
  var superData: Int = 17
}

class Point3D : PointSuperclass, PointUtils {
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

func takes_protocol(_ loc2d : PointUtils,_ loc3d : PointUtils,
  _ loc3dCB : AnyObject & PointUtils,
  _ loc3dSuper : PointSuperclass & PointUtils
) {
    let sum2d = loc2d.sumOfCoordinates()
    let sum3d = loc3d.sumOfCoordinates()
    let sum3dCB = loc3dCB.sumOfCoordinates()
    let sum3dSuper = loc3dSuper.sumOfCoordinates()
    print("hello \(sum2d) \(sum3d) \(sum3dSuper)") // Set breakpoint here
}

func main() {

    var loc2d = Point2D(1.25, 2.5)
    var loc3d = Point3D(1.25, 2.5, 1.25)

  takes_protocol (loc2d, loc3d, loc3d, loc3d)
}

main()

