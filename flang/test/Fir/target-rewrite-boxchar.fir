// RUN: fir-opt --target-rewrite="target=i386-unknown-linux-gnu" %s | FileCheck %s --check-prefix=INT32
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=INT64
// RUN: fir-opt --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=INT64
// RUN: fir-opt --target-rewrite="target=powerpc64le-unknown-linux-gnu" %s | FileCheck %s --check-prefix=INT64

// Test that we rewrite the signatures and bodies of functions that take boxchar
// parameters.
// INT32-LABEL: @boxcharparams
// INT32-SAME: ([[ARG0:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG1:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG2:%[0-9A-Za-z]+]]: i32, [[ARG3:%[0-9A-Za-z]+]]: i32) -> i64
// INT64-LABEL: @boxcharparams
// INT64-SAME: ([[ARG0:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG1:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG2:%[0-9A-Za-z]+]]: i64, [[ARG3:%[0-9A-Za-z]+]]: i64) -> i64
func.func @boxcharparams(%arg0 : !fir.boxchar<1>, %arg1 : !fir.boxchar<1>) -> i64 {
  // INT32-DAG: [[B0:%[0-9]+]] = fir.emboxchar [[ARG1]], [[ARG3]] : (!fir.ref<!fir.char<1,?>>, i32) -> !fir.boxchar<1>
  // INT32-DAG: [[B1:%[0-9]+]] = fir.emboxchar [[ARG0]], [[ARG2]] : (!fir.ref<!fir.char<1,?>>, i32) -> !fir.boxchar<1>
  // INT32-DAG: fir.unboxchar [[B0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  // INT32-DAG: fir.unboxchar [[B1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  // INT64-DAG: [[B0:%[0-9]+]] = fir.emboxchar [[ARG1]], [[ARG3]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT64-DAG: [[B1:%[0-9]+]] = fir.emboxchar [[ARG0]], [[ARG2]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT64-DAG: fir.unboxchar [[B0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  // INT64-DAG: fir.unboxchar [[B1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  %3 = arith.addi %1#1, %2#1 : i64
  return %3 : i64
}

// Test that we rewrite the signatures and bodies of functions that return a
// boxchar.
// INT32-LABEL: @boxcharsret
// INT32-SAME: ([[ARG0:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>> {llvm.sret = !fir.char<1,?>}, [[ARG1:%[0-9A-Za-z]+]]: i32, [[ARG2:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG3:%[0-9A-Za-z]+]]: i32)
// INT64-LABEL: @boxcharsret
// INT64-SAME: ([[ARG0:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>> {llvm.sret = !fir.char<1,?>}, [[ARG1:%[0-9A-Za-z]+]]: i64, [[ARG2:%[0-9A-Za-z]+]]: !fir.ref<!fir.char<1,?>>, [[ARG3:%[0-9A-Za-z]+]]: i64)
func.func @boxcharsret(%arg0 : !fir.boxchar<1> {llvm.sret = !fir.char<1,?>}, %arg1 : !fir.boxchar<1>) {
  // INT32-DAG: [[B0:%[0-9]+]] = fir.emboxchar [[ARG0]], [[ARG1]] : (!fir.ref<!fir.char<1,?>>, i32) -> !fir.boxchar<1>
  // INT32-DAG: [[B1:%[0-9]+]] = fir.emboxchar [[ARG2]], [[ARG3]] : (!fir.ref<!fir.char<1,?>>, i32) -> !fir.boxchar<1>
  // INT32-DAG: fir.unboxchar [[B0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  // INT32-DAG: fir.unboxchar [[B1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  // INT64-DAG: [[B0:%[0-9]+]] = fir.emboxchar [[ARG0]], [[ARG1]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT64-DAG: [[B1:%[0-9]+]] = fir.emboxchar [[ARG2]], [[ARG3]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT64-DAG: fir.unboxchar [[B0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  // INT64-DAG: fir.unboxchar [[B1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %3 = fir.convert %1#1 : (i64) -> index
  %last = arith.subi %3, %c1 : index
  fir.do_loop %i = %c0 to %last step %c1 {
    %in_pos = fir.coordinate_of %2#0, %i : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    %out_pos = fir.coordinate_of %1#0, %i : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    %ch = fir.load %in_pos : !fir.ref<!fir.char<1>>
    fir.store %ch to %out_pos : !fir.ref<!fir.char<1>>
  }
  return
}

// Test that we rewrite the signatures of functions with a sret parameter and
// several other parameters.
// INT32-LABEL: @boxcharmultiple
// INT32-SAME: ({{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>> {llvm.sret = !fir.char<1,?>}, {{%[0-9A-Za-z]+}}: i32, {{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>>, {{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>>, {{%[0-9A-Za-z]+}}: i32, {{%[0-9A-Za-z]+}}: i32)
// INT64-LABEL: @boxcharmultiple
// INT64-SAME: ({{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>> {llvm.sret = !fir.char<1,?>}, {{%[0-9A-Za-z]+}}: i64, {{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>>, {{%[0-9A-Za-z]+}}: !fir.ref<!fir.char<1,?>>, {{%[0-9A-Za-z]+}}: i64, {{%[0-9A-Za-z]+}}: i64)
func.func @boxcharmultiple(%arg0 : !fir.boxchar<1> {llvm.sret = !fir.char<1,?>}, %arg1 : !fir.boxchar<1>, %arg2 : !fir.boxchar<1>) {
  return
}

// Test that we rewrite calls to functions that take boxchar arguments.
// INT32-LABEL: @boxcharcallee(!fir.ref<!fir.char<1,?>>, i32)
// INT64-LABEL: @boxcharcallee(!fir.ref<!fir.char<1,?>>, i64)
func.func private @boxcharcallee(%x : !fir.boxchar<1>)

// INT32: @boxchararg
// INT64: @boxchararg
func.func @boxchararg() {
  %1 = fir.address_of (@name) : !fir.ref<!fir.char<1,9>>
  %2 = arith.constant 9 : i64
  %3 = fir.convert %1 : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<!fir.char<1,?>>
  // INT32: [[B:%[0-9A-Za-z]+]] = fir.emboxchar {{.*}} : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT32: [[U:%[0-9A-Za-z]+]]:2 = fir.unboxchar [[B]] : (!fir.boxchar<1>) ->
  // (!fir.ref<!fir.char<1, ?>>, i32)
  // INT64: [[B:%[0-9A-Za-z]+]] = fir.emboxchar {{.*}} : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT64: [[U:%[0-9A-Za-z]+]]:2 = fir.unboxchar [[B]] : (!fir.boxchar<1>) ->
  // (!fir.ref<!fir.char<1, ?>>, i64)
  %4 = fir.emboxchar %3, %2 : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  // INT32: fir.call @boxcharcallee([[U]]#0, [[U]]#1) : (!fir.ref<!fir.char<1,?>>, i32) -> ()
  // INT64: fir.call @boxcharcallee([[U]]#0, [[U]]#1) : (!fir.ref<!fir.char<1,?>>, i64) -> ()
  fir.call @boxcharcallee(%4) : (!fir.boxchar<1>) -> ()
  return
}

fir.global @name constant : !fir.char<1,9> {
  %str = fir.string_lit "Your name"(9) : !fir.char<1,9>
  //constant 1
  fir.has_value %str : !fir.char<1,9>
}

// Test that we rewrite the fir.address_of operator
// INT32-LABEL: @addrof
// INT64-LABEL: @addrof
func.func @addrof() {
  // INT32: {{.*}} = fir.address_of(@boxcharcallee) : (!fir.ref<!fir.char<1,?>>, i32) -> ()
  // INT64: {{.*}} = fir.address_of(@boxcharcallee) : (!fir.ref<!fir.char<1,?>>, i64) -> ()
  %f = fir.address_of(@boxcharcallee) : (!fir.boxchar<1>) -> ()
  return
}
