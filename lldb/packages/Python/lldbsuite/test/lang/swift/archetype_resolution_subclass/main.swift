class Patatino {}
class Tinky : Patatino {}

// Let's try to make sure that frame var prints the dynamic type, i.e.
// the subclass `Tinky`.
func f<T>(_ arg : T) -> T {
  return arg //%self.expect('frame variable -d run -- arg', substrs=['Tinky'])
}

var x : Patatino = Tinky()
f(x)
print(x)
