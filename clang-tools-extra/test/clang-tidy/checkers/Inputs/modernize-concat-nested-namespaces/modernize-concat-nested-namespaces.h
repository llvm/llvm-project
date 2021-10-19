namespace nn1 {
namespace nn2 {
// CHECK-FIXES: namespace nn1::nn2
void t();
} // namespace nn2
} // namespace nn1
// CHECK-FIXES: void t();
// CHECK-FIXES-NEXT: } // namespace nn1
