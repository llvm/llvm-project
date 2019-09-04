struct A {
    private class X {
        let xx: Int

        init(xx: Int) {
            self.xx = xx
        }
    }
    private class Y {
        let yy: Int

        init(yy: Int) {
            self.yy = yy
        }
    }

    private enum Variant {
        case x(X)
        case y(Y)
        case z
    }

    private let variant: Variant

    init(x: Int) {
        variant = .x(X(xx: x))
    }

    init(y: Int) {
        variant = .y(Y(yy: y))
    }

    init(z: Void) {
        variant = .z
    }
}

let x = A(x: 42)
let y = A(y: 39)
print("!")  //%self.expect("frame var -d run-target -- x", substrs=['(xx = 42)'])
            //%self.expect("expr -d run-target -- x", substrs=['(xx = 42)'])
            //%self.expect("frame var -d run-target -- y", substrs=['(yy = 39)'])
            //%self.expect("expr -d run-target -- y", substrs=['(yy = 39)'])
