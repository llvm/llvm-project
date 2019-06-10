enum BlendMode {
    case opaque
    case trans // this is premul alpha. For additive, just output .a = 1 in the shader
    case mask(cutoff: Float)
}

var blend = BlendMode.trans
blend = .mask(cutoff: 0.5) //%self.expect('frame var -d run-target -- blend', substrs=['trans'])
