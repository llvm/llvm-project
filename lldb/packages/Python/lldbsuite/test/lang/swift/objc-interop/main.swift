import Dispatch

// The dispatch clang module is either imported as Objective-C or as C
// with blocks, and both varints should work in LLDB's expression evaluator.

func main() {
  let label = "lldbtest"
  let queue = DispatchQueue(label: label)
  print(queue) //% self.expect("fr var -- label", substrs=['lldbtest'])
               //% self.expect("expr -- label",   substrs=['lldbtest'])
}

main()
