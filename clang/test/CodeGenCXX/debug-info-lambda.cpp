// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm \
// RUN:   -debug-info-kind=line-tables-only -std=c++11 %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}}lambda_in_func
void lambda_in_func(int &ref) {
  // CHECK: [[ref_slot:%.*]] = getelementptr inbounds %class.anon, %class.anon* {{.*}}, i32 0, i32 0, !dbg [[lambda_decl_loc:![0-9]+]]
  // CHECK-NEXT: %1 = load i32*, i32** %ref.addr, align {{.*}}, !dbg [[capture_init_loc:![0-9]+]]
  // CHECK-NEXT: store i32* %1, i32** %0, align {{.*}}, !dbg [[lambda_decl_loc]]
  // CHECK-NEXT: call {{.*}}void {{.*}}, !dbg [[lambda_call_loc:![0-9]+]]

  auto helper = [       // CHECK: [[lambda_decl_loc]] = !DILocation(line: [[@LINE]], column: 17
                 &]() { // CHECK: [[capture_init_loc]] = !DILocation(line: [[@LINE]], column: 18
    ++ref;
  };
  helper();             // CHECK: [[lambda_call_loc]] = !DILocation(line: [[@LINE]]
}
