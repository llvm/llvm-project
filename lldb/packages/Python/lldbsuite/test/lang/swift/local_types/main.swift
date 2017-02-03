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

    struct Foo {
        init () {
            a = 234
            b = 1.25
        }
        var a : Int;
        var b : Double;

        struct Bar {
            init () {
                c = 48
                d = "Hello"
            }
            var c: Int8;
            var d: String;
        }
    }
    
    class Base {
      var a = 1
    }
    
    class Derived : Base {
      var b = 2
    }

    var f = Foo()
    var b = Foo.Bar()
    var c = Derived()
    print("I like my f") //% self.expect("frame variable f", substrs = ['f = (a = 234, b = 1.25)'])
    print("and my b too") //% self.expect("frame variable b", substrs = ['b = (c = 48, d = "Hello")'])
    print("and my c is quite cool") //% self.expect("frame variable -d run -- c", substrs = ['a = 1', 'b = 2'])
    print("[chorus] I like my f") //% self.expect("expr f", substrs = ['= (a = 234, b = 1.25)'])
    print("[chorus] and my b too") //% self.expect("expr b", substrs = ['= (c = 48, d = "Hello")'])
    print("[chorus] and my c is quite cool") //% self.expect("expr -d run -- c", substrs = ['a = 1', 'b = 2'])
    return 0
}

main()
