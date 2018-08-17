// This test contains a subset of Swift that is expected to work in
// LLDB even when compiled with (slightly) older versions of the Swift
// compiler.

func use<T>(_ t: T) {}

func f<T>(_ t: T) {
  let number = 23
  let array = [1, 2, 3]
  let string = "hello"
  let tuple = (42, "abc")
  let generic = t
  print(number) // break here
  use(number)
  use(array)
  use(string)
  use(tuple)
  use(generic)
}

f(-1)
