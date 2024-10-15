! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module m
integer XcX
end


CHECK: !DIGlobalVariable(name: "xcx", linkageName: "_QMmExcx"{{.*}})
