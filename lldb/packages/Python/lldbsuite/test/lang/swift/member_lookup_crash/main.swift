import Foundation

extension Measurement where UnitType == UnitAngle {
  var radians: CGFloat {
    return CGFloat(self.converted(to: .radians).value)
  }

  var degrees: CGFloat {
    return CGFloat(self.converted(to: .degrees).value)
  }

  func f() {
    return //%self.expect('p self.radians', substrs=['is not convertible to'], error=True)
  }
}

let measure = Measurement<UnitAngle>(value: 100, unit: UnitAngle.degrees)
measure.f()
