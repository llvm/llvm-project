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
class DefaultMirror {
  var a = "y12"
  var b = "q24"
}

class CustomMirror : CustomReflectable {
  var a = 12
  var b = 24
  
  public var customMirror: Mirror {
    get { return Mirror(self, children: ["c" : "t\(a + b)"]) }
  }
}

class CustomSummary : CustomStringConvertible, CustomDebugStringConvertible {
  var a = 12
  var b = 24
  
  var description: String { return "CustomStringConvertible" }
  var debugDescription: String { return "CustomDebugStringConvertible" }
}

class TheBase : CustomReflectable {
  var a = "y12"
  var b = "q24"
  
  public var customMirror: Mirror {
    get { return Mirror(self, children: ["a" : a, b : "b"], displayStyle: .`class`) }
  }
}

class TheDescendant : TheBase {
  var c = "t36"
}

class TheReflectiveDescendant: TheBase {
  var d = "w48"
  
  public override var customMirror: Mirror {
    get { return Mirror(self, children: ["d" : d], displayStyle: .`class`) }
  }
}

func main() {
  var dm = DefaultMirror()
  var cm = CustomMirror()
  var cs = CustomSummary()
  var td = TheDescendant()
  var tr = TheReflectiveDescendant()
  print("yay I am done!") //% self.expect("po dm", substrs=['y12', 'q24'], matching=False)
  //% self.expect("po dm", substrs=['DefaultMirror: 0x'])
  //% self.expect("po cm", substrs=['t36'])
  //% self.expect("po cm", substrs=['y12', 'q24'], matching=False)
  //% self.expect("po cs", substrs=['CustomDebugStringConvertible'])
  //% self.expect("po cs", substrs=['CustomStringConvertible'], matching=False)
  //% self.expect("po cs", substrs=['y12', 'q24'], matching=False)
  //% self.expect("po [dm as Any,cm as Any,48 as Any]", substrs=['t36', '48'])
  //% self.expect("po [dm as Any,cm as Any,48 as Any]", substrs=['y12', 'q24'], matching=False)
  //% self.expect("po td", substrs=['TheDescendant', 'y12','q24'])
  //% self.expect("po td", substrs=['t36'], matching=False)
  //% self.expect("po tr", substrs=['TheReflectiveDescendant', 'w48', 'q24', 'y12'])
  //% self.expect("po tr", substrs=['t36'], matching=False)
  //% self.expect("script lldb.frame.FindVariable('tr').GetObjectDescription()", substrs=['t36'], matching=False)
  //% self.expect("script lldb.frame.FindVariable('tr').GetObjectDescription()", substrs=['TheReflectiveDescendant', 'w48', 'q24', 'y12'])
}

main()
