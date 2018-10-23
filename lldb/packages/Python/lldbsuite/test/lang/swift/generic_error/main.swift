enum MyErr : Error {
  case Patatino
  case Mio
}

func f<T>(_ Pat : T) -> T {
  return Pat //%self.expect("frame var -d run-target -- Pat", substrs=['(a.MyErr) Pat = Patatino'])
}

let patatino = f(MyErr.Patatino as Error)
print(patatino)
