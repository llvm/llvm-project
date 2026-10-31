// RUN: inter-opt --split-input-file -verify-diagnostics %s

func.func @sub_grf_tuple(%lo: !xemachine.reg<8, -1>,
                         %hi: !xemachine.reg<8, -1>) {
  // expected-error @+1 {{'xemachine.tuple_from_elements' op tuple elements must occupy whole 16-dword GRFs}}
  %tuple = xemachine.tuple_from_elements %lo, %hi
      : (!xemachine.reg<8, -1>, !xemachine.reg<8, -1>)
      -> !xemachine.reg<16, -1>
  return
}

// -----

func.func @sub_grf_split(%tuple: !xemachine.reg<16, -1>) {
  // expected-error @+1 {{'xemachine.tuple_to_elements' op tuple elements must occupy whole 16-dword GRFs}}
  %lo, %hi = xemachine.tuple_to_elements %tuple
      : (!xemachine.reg<16, -1>)
      -> (!xemachine.reg<8, -1>, !xemachine.reg<8, -1>)
  return
}
