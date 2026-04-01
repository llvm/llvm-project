func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x12x8xf32>, %filter: tensor<3x8x8xf32>)
// <- function.builtin
//        ^ function
//                                   ^ variable.parameter
//                                           ^ type.builtin
//                                                               ^ variable.parameter
//                                                                        ^ type.builtin
  -> tensor<1x10x8x8xf32> {
// ^ operator
//   ^ type.builtin
  %zero = arith.constant 0.000000e+00 : f32
// ^ variable
//        ^ function.builtin
//                       ^ number
//                                      ^ type.builtin
  %init = tensor.empty() : tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                         ^ type.builtin
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                    ^ keyword
//                        ^ variable
//                                ^ type.builtin
//                                     ^ keyword
  %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>,
// ^ variable
//     ^ function.builtin
//                                       ^ attribute
//                                                   ^ constant.builtin
    strides = dense<1> : tensor<1xi64>}
//            ^ constant.builtin
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8x8xf32>)
//      ^ variable.parameter
//              ^ variable.parameter
    outs(%fill : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
//       ^ variable
  return %0 : tensor<1x10x8x8xf32>
// ^ function.builtin
//       ^ variable
}

func.func @fastmath(%arg0: f32, %arg1: f32) {
// <- function.builtin
//        ^ function
//                  ^ variable.parameter
//                         ^ type.builtin
//                              ^ variable.parameter
//                                     ^ type.builtin
  %5 = arith.negf %arg0 fastmath<fast> : f32
//     ^ function.builtin
//                      ^ attribute
  %6 = arith.addf %arg0, %arg1 fastmath<none> : f32
//     ^ function.builtin
//                             ^ attribute
  %8 = arith.mulf %arg0, %arg1 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
//     ^ function.builtin
//                             ^ attribute
  return
// ^ function.builtin
}

#map0 = affine_map<(d0, d1) -> (d0, d1)>
// <- attribute
//      ^ attribute
#map1 = affine_map<(d0, d1) -> (d0)>
// <- attribute
//      ^ attribute
#map2 = affine_map<(d0) -> (d0)>
// <- attribute
//      ^ attribute

func.func @add_broadcast_mul_fusion(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>,
  %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.empty(%0) : tensor<?xf32>
  %2 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]}
//                                      ^ attribute
//                                             ^ attribute
//                                                    ^ attribute
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
//    ^ keyword
      outs(%1 : tensor<?xf32>) {
//    ^ keyword
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = arith.addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  %3 = tensor.dim %arg2, %c1 : tensor<?x?xf32>
  %4 = tensor.empty(%0, %3) : tensor<?x?xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
//     ^ function.builtin
      ins(%2, %arg2 : tensor<?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>){
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
      %6 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

func.func @broadcast(%input: tensor<8x32xf32>,
                     %init: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %bcast = linalg.broadcast
//         ^ function.builtin
      ins(%input:tensor<8x32xf32>)
//    ^ keyword
      outs(%init:tensor<8x16x32xf32>)
//    ^ keyword
      dimensions = [1]
//    ^ attribute
  func.return %bcast : tensor<8x16x32xf32>
}
