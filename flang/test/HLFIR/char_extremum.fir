// Test hlfir.char_extremum operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s


// variable check
func.func @char_extremum_min_var(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>) { 
  %0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>
  return
}

// CHECK-LABEL:   func.func @char_extremum_min_var(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,20>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>

func.func @char_extremum_max_var(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>) { 
%0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
return
}

// CHECK-LABEL:   func.func @char_extremum_max_var(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,20>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>

func.func @char_extremum_min_var_variadic(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>, %arg2: !fir.ref<!fir.char<1,30>>) { 
  %0 = hlfir.char_extremum min, %arg0, %arg1, %arg2 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,10>>
  return
}

// CHECK-LABEL:   func.func @char_extremum_min_var_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,20>>,
// CHECK-SAME:    %[[VAL_2:.*]]: !fir.ref<!fir.char<1,30>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,10>>

func.func @char_extremum_max_var_variadic(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>, %arg2: !fir.ref<!fir.char<1,30>>) { 
  %0 = hlfir.char_extremum max, %arg0, %arg1, %arg2 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,30>>
  return
}

// CHECK-LABEL:   func.func @char_extremum_max_var_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.ref<!fir.char<1,20>>,
// CHECK-SAME:    %[[VAL_2:.*]]: !fir.ref<!fir.char<1,30>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum max, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,30>>


// boxchar check
func.func @char_extremum_min_boxchar(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
 %0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_extremum_min_boxchar(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.boxchar<1>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]] : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>

func.func @char_extremum_max_boxchar(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
 %0 = hlfir.char_extremum max, %arg0, %arg1 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_extremum_max_boxchar(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.boxchar<1>) {
// CHECK:  %{{.*}} = hlfir.char_extremum max, %[[VAL_0]], %[[VAL_1]] : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>

func.func @char_extremum_min_boxchar_variadic(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) { 
  %0 = hlfir.char_extremum min, %arg0, %arg1, %arg2, %arg3 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_extremum_min_boxchar_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.boxchar<1>, 
// CHECK-SAME:    %[[VAL_2:[a-zA-Z0-9_]*]]: !fir.boxchar<1>, 
// CHECK-SAME:    %[[VAL_3:[a-zA-Z0-9_]*]]: !fir.boxchar<1>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>

func.func @char_extremum_max_boxchar_variadic(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) { 
  %0 = hlfir.char_extremum max, %arg0, %arg1, %arg2, %arg3 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_extremum_max_boxchar_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !fir.boxchar<1>, 
// CHECK-SAME:    %[[VAL_2:[a-zA-Z0-9_]*]]: !fir.boxchar<1>, 
// CHECK-SAME:    %[[VAL_3:[a-zA-Z0-9_]*]]: !fir.boxchar<1>) {
// CHECK:  %{{.*}} = hlfir.char_extremum max, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>


//expr check
func.func @char_extremum_min_expr(%arg0: !hlfir.expr<!fir.char<1,10>>, %arg1: !hlfir.expr<!fir.char<1,20>>) {
  %0 = hlfir.char_extremum min, %arg0, %arg1 : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>) -> (!hlfir.expr<!fir.char<1,10>>)
  return
}
// CHECK-LABEL:   func.func @char_extremum_min_expr(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,20>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>

func.func @char_extremum_max_expr(%arg0: !hlfir.expr<!fir.char<1,10>>, %arg1: !hlfir.expr<!fir.char<1,20>>) {
  %0 = hlfir.char_extremum max, %arg0, %arg1 : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>) -> (!hlfir.expr<!fir.char<1,20>>)
  return
}
// CHECK-LABEL:   func.func @char_extremum_max_expr(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,20>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum max, %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>

func.func @char_extremum_min_expr_variadic(%arg0: !hlfir.expr<!fir.char<1,10>>, %arg1: !hlfir.expr<!fir.char<1,20>>, %arg2: !hlfir.expr<!fir.char<1,30>>) {
  %0 = hlfir.char_extremum min, %arg0, %arg1, %arg2 : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, !hlfir.expr<!fir.char<1,30>>) -> (!hlfir.expr<!fir.char<1,10>>)
  return
}
// CHECK-LABEL:   func.func @char_extremum_min_expr_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,20>>,
// CHECK-SAME:    %[[VAL_2:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,30>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum min, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, !hlfir.expr<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,10>>

func.func @char_extremum_max_expr_variadic(%arg0: !hlfir.expr<!fir.char<1,10>>, %arg1: !hlfir.expr<!fir.char<1,20>>, %arg2: !hlfir.expr<!fir.char<1,30>>) {
  %0 = hlfir.char_extremum max, %arg0, %arg1, %arg2 : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, !hlfir.expr<!fir.char<1,30>>) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}
// CHECK-LABEL:   func.func @char_extremum_max_expr_variadic(
// CHECK-SAME:    %[[VAL_0:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,20>>,
// CHECK-SAME:    %[[VAL_2:[a-zA-Z0-9_]*]]: !hlfir.expr<!fir.char<1,30>>) {
// CHECK:  %{{.*}} = hlfir.char_extremum max, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, !hlfir.expr<!fir.char<1,30>>) -> !hlfir.expr<!fir.char<1,30>>

