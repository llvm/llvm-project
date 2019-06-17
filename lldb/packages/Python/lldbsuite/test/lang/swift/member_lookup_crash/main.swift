import Foundation

extension Measurement where UnitType == UnitAngle {
  var radians: CGFloat {
    return CGFloat(self.converted(to: .radians).value)
  }

  var degrees: CGFloat {
    return CGFloat(self.converted(to: .degrees).value)
  }

  func f() {
    return //%self.expect('p self.radians', substrs=["(CGFloat) $R0 = 1.745"])
           //%self.expect('p self', substrs=["Measurement<UnitAngle>"])
  }
}

let measure = Measurement<UnitAngle>(value: 100, unit: UnitAngle.degrees)
measure.f()
