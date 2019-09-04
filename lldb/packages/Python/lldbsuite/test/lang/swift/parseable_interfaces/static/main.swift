import AA
import BB

func foo() {
  let x = MyPoint(x: 10, y: 0)
  let y = MyPoint(x: -2, y: 20)

  print(x.magnitudeSquared)
  print(y.magnitudeSquared)

  Bar.bar() // break here
}

foo()
