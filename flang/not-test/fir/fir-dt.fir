func @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)

fir.dispatch_table @dispatch_tbl {
  fir.dt_entry method, @method_impl
}
