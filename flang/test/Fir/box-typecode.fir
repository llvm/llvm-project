// RUN: tco %s | FileCheck %s

func.func @test_box_typecode(%a: !fir.class<none>) -> i32 {
  %0 = fir.box_typecode %a : (!fir.class<none>) -> i32
  return %0 : i32
}

// CHECK-LABEL: @test_box_typecode(
// CHECK-SAME: ptr %[[BOX:.*]]) 
// CHECK: %[[GEP:.*]] = getelementptr { ptr, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}} }, ptr %[[BOX]], i32 0, i32 4
// CHECK: %[[TYPE_CODE:.*]] = load i8, ptr %[[GEP]]
// CHECK: %[[TYPE_CODE_CONV:.*]] = sext i8 %[[TYPE_CODE]] to i32
// CHECK: ret i32 %[[TYPE_CODE_CONV]]
