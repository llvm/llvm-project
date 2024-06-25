// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void foo(int* v) {
}

void m1(int* v1, int* v2, int) {

}

void condition_check_nullptr() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  if(p != nullptr) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
}

void condition_check() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  auto q = new int[10];

  int tmp = p[5];
  if(p == q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(q != p){}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:12}:".data()"

  if(p < q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p <= q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p > q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if(p >= q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  if( p == q && p != nullptr) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:".data()"
}

void condition_check_two_usafe_buffers() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int* q = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  tmp = q[5];

  if(p == q) {}
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:".data()"
}

void unsafe_method_invocation_single_param() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  foo(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"

}

void unsafe_method_invocation_single_param_array(int idx) {
  int p[32];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::array<int, 32> p"

  int tmp = p[idx];
  foo(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"
}

void safe_method_invocation_single_param() {
  int* p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}
  foo(p);
}

void safe_method_invocation_single_param_array() {
  int p[10];
  foo(p);
  // CHECK-NO: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}:".data()"
}

void unsafe_method_invocation_double_param() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  m1(p, p, 10);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:".data()"

  auto q = new int[10];

  m1(p, q, 4);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"

  m1(q, p, 9);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:10}:".data()"

  m1(q, q, 8);
}

void unsafe_method_invocation_double_param_array(int idx) {
  int p[14];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::array<int, 14> p"

  int q[40];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::array<int, 40> q"

  q[idx] = p[idx];

  m1(p, p, 10);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:".data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:".data()"
}

void unsafe_access_in_lamda() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  auto my_lambda = [&](){
    p[5] = 10;
  };

  foo(p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:".data()"
}

void fixits_in_lamda() {
  int* p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  auto my_lambda = [&](){
    foo(p);
    // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:10}:".data()"
  };

  p[5] = 10;
}

// FIXME: Emit fixits for lambda captured variables
void fixits_in_lambda_capture() {
  auto p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  
  auto my_lambda = [p](){ // No fixits emitted here.
    foo(p);
  };
  
  p[5] = 10;
}

void fixits_in_lambda_capture_reference() {
  auto p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
 
  auto my_lambda = [&p](){ // No fixits emitted here.
    foo(p);
  };

  p[5] = 10;
}

void fixits_in_lambda_capture_rename() {
  auto p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  
  auto my_lambda = [x = p](){ // No fixits emitted here.
    foo(x);
  };

  p[5] = 10;
}

bool ptr_comparison(int* ptr, unsigned idx) {
  int arr[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::array<int, 10> arr"
  arr[idx] = idx;

  return arr > ptr;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:".data()"
}

int long long ptr_distance(int* ptr, unsigned idx) {
  int arr[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::array<int, 10> arr"
  arr[idx] = idx;

  int long long dist = arr - ptr;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:27-[[@LINE-1]]:27}:".data()"
  return dist;
}
