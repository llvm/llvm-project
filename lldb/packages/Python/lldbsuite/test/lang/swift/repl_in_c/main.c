// main.c
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

int main()
{
    return 0; //% self.expect("repl", error=True, substrs=["Swift standard library"])
              //% self.runCmd("kill")
              //% self.expect("repl", error=True, substrs=["running process"])
}

