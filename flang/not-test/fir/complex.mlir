func @add(%a : complex<f32>, %b : complex<f32>) -> complex<f32>

func @foo(%a : complex<f32>, %b : complex<f32>) -> complex<f32> {
  %1 = call @add(%a, %b) : (complex<f32>, complex<f32>) -> complex<f32>
  return %1 : complex<f32>
}
