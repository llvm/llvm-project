import AA

func foo() {
  let x = MyPoint(x: 10, y: 0)
  let y = MyPoint(x: -2, y: 20)

  print("x=\(x.magnitudeSquared)")
  print("y=\(y.magnitudeSquared)")
}

foo()
