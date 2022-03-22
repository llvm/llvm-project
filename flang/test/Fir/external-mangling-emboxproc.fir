// RUN: fir-opt --external-name-interop %s | FileCheck %s

func @_QPfoo() {  
  %e6 = fir.alloca tuple<i32,f64>
  %ao = fir.address_of(@_QPfoo_impl) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()
  %0 = fir.emboxproc %ao, %e6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  return
}
func private @_QPfoo_impl(!fir.ref<i32>)

// CHECK: fir.address_of(@foo_impl_)
