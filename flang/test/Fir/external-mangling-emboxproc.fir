// RUN: fir-opt --external-name-interop %s | FileCheck %s

func @_QPfoo() {  
  %e6 = fir.alloca tuple<i32,f64>
  %0 = fir.emboxproc @_QPfoo_impl, %e6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>  
  return
}
func private @_QPfoo_impl(!fir.ref<i32>)

// CHECK: %{{.*}}= fir.emboxproc @foo_impl_
