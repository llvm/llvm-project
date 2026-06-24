// RUN: %clang_cc1 -emit-llvm -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -o- %s | FileCheck %s

#line 100
int main() {
  try {
    throw 0;
  } catch (bool) {
    return 0;
  } catch (int) {
    return 1;
  }
}

// CHECK:        %{{.*}} = landingpad { ptr, i32 }
// CHECK-NEXT:           catch ptr @_ZTIb
// CHECK-NEXT:           catch ptr @_ZTIi, !dbg [[LPAD:![0-9]*]]
// CHECK:        br label %[[DISP:.*]], !dbg [[LPAD]]
// CHECK:      [[DISP]]:
// CHECK-NEXT:   %{{.*}} = load i32, ptr %{{.*}}, align 4, !dbg [[DISPATCH:![0-9]*]]
// CHECK-NEXT:   [[TIBOOL:%.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIb) #{{[0-9]+}}, !dbg [[DISPATCH]]
// CHECK-NEXT:   [[ISBOOL:%.*]] = icmp eq i32 %{{.*}}, [[TIBOOL]], !dbg [[DISPATCH]]
// CHECK-NEXT:   br i1 [[ISBOOL]], label %[[CATCHBOOL:.*]], label %[[FALLTHROUGH:.*]], !dbg [[DISPATCH]]
// CHECK:      [[CATCHBOOL]]
// CHECK:        %{{.*}} = call ptr @__cxa_begin_catch(ptr %{{.*}}) #{{[0-9]+}}, !dbg [[BEGINCATCHBOOL:![0-9]*]]
// CHECK:        call void @__cxa_end_catch() #{{[0-9]+}}, !dbg [[ENDCATCHBOOL:![0-9]*]]
// CHECK:      [[FALLTHROUGH]]:
// CHECK-NEXT:   [[TIINT:%.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi) #{{[0-9]+}}, !dbg [[DISPATCH]]
// CHECK-NEXT:   [[ISINT:%.*]] = icmp eq i32 %{{.*}}, [[TIINT]], !dbg [[DISPATCH]]
// CHECK-NEXT:   br i1 [[ISINT]], label %[[CATCHINT:.*]], label %[[RESUME:.*]], !dbg [[DISPATCH]]
// CHECK:      [[CATCHINT]]:
// CHECK:        %{{.*}} = call ptr @__cxa_begin_catch(ptr %{{.*}}) #{{[0-9]+}}, !dbg [[BEGINCATCHINT:![0-9]*]]
// CHECK:        call void @__cxa_end_catch() #{{[0-9]+}}, !dbg [[ENDCATCHINT:![0-9]*]]
// CHECK:      [[RESUME]]:
// CHECK:        resume { ptr, i32 } %{{.*}}, !dbg [[DISPATCH]]

// TODO: LPAD is arguably off.
// CHECK-DAG: [[LPAD]] = !DILocation(line: 108, column: 1, scope: !{{[0-9]+}})
// CHECK-DAG: [[DISPATCH]] = !DILocation(line: 103, column: 3, scope: !{{[0-9]+}})
// CHECK-DAG: [[BEGINCATCHBOOL]] = !DILocation(line: 103, column: 18, scope: !{{[0-9]+}})
// CHECK-DAG: [[ENDCATCHBOOL]] = !DILocation(line: 105, column: 3, scope: !{{[0-9]+}})
// CHECK-DAG: [[BEGINCATCHINT]] = !DILocation(line: 105, column: 17, scope: !{{[0-9]+}})
// CHECK-DAG: [[ENDCATCHINT]] = !DILocation(line: 107, column: 3, scope: !{{[0-9]+}})
