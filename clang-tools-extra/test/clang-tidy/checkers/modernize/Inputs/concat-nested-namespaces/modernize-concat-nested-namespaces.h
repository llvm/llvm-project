// CHECK-MESSAGES-NORMAL: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace nn1 {
namespace nn2 {
// CHECK-FIXES-NORMAL: namespace nn1::nn2 {
void t();
} // namespace nn2
} // namespace nn1
// CHECK-FIXES-NORMAL: void t();
// CHECK-FIXES-NORMAL-NEXT: } // namespace nn1::nn2
